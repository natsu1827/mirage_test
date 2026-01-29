import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tempfile
import shutil
from pathlib import Path
from mutils.gcs_utils import download_file_from_gcs, upload_file_to_gcs, list_gcs_files

# ==========================================
# 設定區
# ==========================================
INPUT_RAW_DIR = os.getenv("INPUT_BUCKET_URI", "gs://oct-image-test-v1")
INPUT_MASK_DIR = os.getenv("MASK_BUCKET_URI", "gs://oct-seg-mask-test-v1")
OUTPUT_DIR = os.getenv("OUTPUT_BUCKET_URI", "gs://oct-image-output-test-v1")

# 模擬前端輸入：若無數據請設為 None，若有數據請設為 float (例如 3.5)
USER_PIXEL_SPACING = None 

# AROI JSON Label 定義
LABEL_SRF = 161    # Mask 6
LABEL_CYST = 115   # Mask 5
LABEL_PED = 138    # Mask 5
LABEL_DRUSEN = 69  # Mask 3

# 色盲友善配色
COLORS = {
    "SRF":    {"hex": "#00FFFF", "rgb": (0, 255, 255)},    # Cyan
    "Cyst":   {"hex": "#FFA500", "rgb": (255, 165, 0)},    # Orange
    "PED":    {"hex": "#FF00FF", "rgb": (255, 0, 255)},    # Magenta
    "Drusen": {"hex": "#FFFF00", "rgb": (255, 255, 0)}     # Yellow
}

def is_gcs(path):
    return path.startswith("gs://")

# Removed redundant list_gcs_files, download_from_gcs, upload_to_gcs
# (Now using imported versions)

class AMDVisualizer:
    def __init__(self, raw_path, mask_path, filename, pixel_spacing=None):
        self.filename = filename
        self.pixel_spacing = pixel_spacing
        
        # 1. 【關鍵】讀取原始影像，並鎖定尺寸為「標準」
        self.raw_image = Image.open(raw_path).convert("RGBA")
        self.width, self.height = self.raw_image.size
        # print(f"Debug: Raw Image Size: {self.width}x{self.height}")
        
        # 2. 建立一個與原圖「尺寸完全相同」的全透明圖層
        self.overlay_image = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        self.draw = ImageDraw.Draw(self.overlay_image)
        
        # 3. 讀取 Mask 並強制調整至原圖尺寸
        mask_img = Image.open(mask_path)
        
        # 如果 Mask 尺寸跟原圖不一樣，強制 Resize Mask (使用 NEAREST 保持 Label 數值)
        if mask_img.size != (self.width, self.height):
            print(f"Warning: Mask size {mask_img.size} mismatch. Resizing to {self.width}x{self.height}")
            mask_img = mask_img.resize((self.width, self.height), resample=Image.NEAREST)
            
        self.mask_array = np.array(mask_img)

        # 儲存結果數據
        self.results = {
            "filename": filename,
            "image_width": self.width,   # 紀錄尺寸供前端參考
            "image_height": self.height,
            "pixel_spacing_um": pixel_spacing,
            "measurements": {}
        }

    def _format_value(self, value_px, type="length"):
        if self.pixel_spacing is None:
            return {"value": round(float(value_px), 2), "unit": "px"}
        
        if type == "length":
            val_um = value_px * self.pixel_spacing
            return {"value": round(val_um, 1), "unit": "um"}
        elif type == "area":
            val_mm2 = value_px * (self.pixel_spacing ** 2) / 1_000_000
            return {"value": round(val_mm2, 4), "unit": "mm2"}

    def draw_contours(self):
        """針對 SRF & Cyst：繪製空心輪廓"""
        targets = [
            (LABEL_SRF, "SRF"),
            (LABEL_CYST, "Cyst")
        ]

        for label_val, name in targets:
            binary_mask = (self.mask_array == label_val).astype(np.uint8) * 255
            pixel_count = np.sum(binary_mask > 0)
            
            if pixel_count > 0:
                self.results["measurements"][name] = {
                    "type": "Area",
                    "data": self._format_value(pixel_count, type="area"),
                    "color": COLORS[name]["hex"]
                }

                # 製作輪廓線
                mask_img = Image.fromarray(binary_mask, mode='L')
                edges = mask_img.filter(ImageFilter.FIND_EDGES)
                
                # 建立純色層 (尺寸確保與原圖一致)
                color_rgb = COLORS[name]["rgb"]
                color_layer = Image.new("RGBA", (self.width, self.height), color_rgb)
                
                # 貼到 overlay_image 上
                self.overlay_image.paste(color_layer, (0, 0), mask=edges)

    def draw_vertical_caliper(self):
        """針對 PED：繪製垂直測量線"""
        label_val = LABEL_PED
        name = "PED"
        y_idxs, x_idxs = np.where(self.mask_array == label_val)
        
        if len(x_idxs) == 0: return

        unique_xs = np.unique(x_idxs)
        max_h = 0
        best_x, best_y1, best_y2 = 0, 0, 0

        for x in unique_xs:
            ys = y_idxs[x_idxs == x]
            h = ys.max() - ys.min()
            if h > max_h:
                max_h = h
                best_x = x
                best_y1 = ys.min()
                best_y2 = ys.max()

        self.results["measurements"][name] = {
            "type": "Max Height",
            "data": self._format_value(max_h, type="length"),
            "color": COLORS[name]["hex"]
        }

        color = COLORS[name]["rgb"]
        lw = 2
        cap = 8
        self.draw.line([(best_x, best_y1), (best_x, best_y2)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y1), (best_x+cap, best_y1)], fill=color, width=lw)
        self.draw.line([(best_x-cap, best_y2), (best_x+cap, best_y2)], fill=color, width=lw)

    def draw_horizontal_caliper(self):
        """針對 Drusen：繪製水平測量線"""
        label_val = LABEL_DRUSEN
        name = "Drusen"
        y_idxs, x_idxs = np.where(self.mask_array == label_val)
        
        if len(x_idxs) == 0: return

        base_y = np.max(y_idxs)
        bottom_pixels = x_idxs[y_idxs >= (base_y - 3)]
        
        if len(bottom_pixels) == 0: return

        min_x, max_x = np.min(bottom_pixels), np.max(bottom_pixels)
        width_px = max_x - min_x
        
        self.results["measurements"][name] = {
            "type": "Max Width",
            "data": self._format_value(width_px, type="length"),
            "color": COLORS[name]["hex"]
        }

        color = COLORS[name]["rgb"]
        draw_y = base_y + 2
        lw = 2
        cap = 5
        self.draw.line([(min_x, draw_y), (max_x, draw_y)], fill=color, width=lw)
        self.draw.line([(min_x, draw_y-cap), (min_x, draw_y+cap)], fill=color, width=lw)
        self.draw.line([(max_x, draw_y-cap), (max_x, draw_y+cap)], fill=color, width=lw)

    def save_results(self):
        base_name = os.path.splitext(self.filename)[0]
        
        # 1. 準備本地存檔路徑
        with tempfile.TemporaryDirectory() as tmp_dir:
            overlay_filename = f"{base_name}_overlay.png"
            analyzed_filename = f"{base_name}_analyzed.png"
            json_filename = f"{base_name}_result.json"
            
            local_overlay_path = os.path.join(tmp_dir, overlay_filename)
            local_analyzed_path = os.path.join(tmp_dir, analyzed_filename)
            local_json_path = os.path.join(tmp_dir, json_filename)

            # 儲存純透明標註圖 (Overlay)
            self.overlay_image.save(local_overlay_path)
            
            # 儲存合成圖 (Combined)
            combined_img = Image.alpha_composite(self.raw_image, self.overlay_image)
            combined_img.convert("RGB").save(local_analyzed_path)
            
            # 儲存 JSON
            self.results["output_image"] = analyzed_filename
            self.results["overlay_image"] = overlay_filename
            with open(local_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=4, ensure_ascii=False)

            # 上傳或移動到最終目的地
            for local_file, filename in [
                (local_overlay_path, overlay_filename),
                (local_analyzed_path, analyzed_filename),
                (local_json_path, json_filename)
            ]:
                if is_gcs(OUTPUT_DIR):
                    target_gcs = f"{OUTPUT_DIR.rstrip('/')}/{filename}"
                    upload_to_gcs(local_file, target_gcs)
                else:
                    if not os.path.exists(OUTPUT_DIR):
                        os.makedirs(OUTPUT_DIR)
                    shutil.copy(local_file, os.path.join(OUTPUT_DIR, filename))
            
            final_out_path = f"{OUTPUT_DIR.rstrip('/')}/{analyzed_filename}"
            return final_out_path, overlay_filename, json_filename

# ==========================================
# 主程式
# ==========================================
def main():
    # 建立輸出目錄 (如果是本地路徑)
    if not is_gcs(OUTPUT_DIR) and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 取得檔案列表
    if is_gcs(INPUT_RAW_DIR):
        raw_files = list_gcs_files(INPUT_RAW_DIR)
    else:
        if not os.path.exists(INPUT_RAW_DIR):
            print(f"Directory not found: {INPUT_RAW_DIR}")
            return
        raw_files = sorted([f for f in os.listdir(INPUT_RAW_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not raw_files:
        print(f"找不到圖片檔案於 {INPUT_RAW_DIR}")
        return

    print(f"開始分析... (Pixel Spacing: {USER_PIXEL_SPACING})")

    for filename in raw_files:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 處理 Raw Image 路徑
                if is_gcs(INPUT_RAW_DIR):
                    local_raw_path = os.path.join(tmp_dir, filename)
                    download_from_gcs(f"{INPUT_RAW_DIR.rstrip('/')}/{filename}", local_raw_path)
                else:
                    local_raw_path = os.path.join(INPUT_RAW_DIR, filename)

                # 處理 Mask 路徑
                if is_gcs(INPUT_MASK_DIR):
                    local_mask_path = os.path.join(tmp_dir, f"mask_{filename}")
                    gcs_mask_url = f"{INPUT_MASK_DIR.rstrip('/')}/{filename}"
                    # 檢查 Mask 是否存在 (gcs)
                    try:
                        download_from_gcs(gcs_mask_url, local_mask_path)
                    except:
                        print(f"[SKIP] 找不到 GCS Mask: {filename}")
                        continue
                else:
                    local_mask_path = os.path.join(INPUT_MASK_DIR, filename)
                    if not os.path.exists(local_mask_path):
                        print(f"[SKIP] 找不到本地 Mask: {filename}")
                        continue
                
                viz = AMDVisualizer(local_raw_path, local_mask_path, filename, pixel_spacing=USER_PIXEL_SPACING)
                
                viz.draw_contours()
                viz.draw_vertical_caliper()
                viz.draw_horizontal_caliper()
                
                img_out, overlay_out, json_out = viz.save_results()
                
                print(f"完成: {filename}")
                print(f"  -> 尺寸: {viz.width}x{viz.height}")
                print(f"  -> 輸出位置: {img_out}")
            
        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()