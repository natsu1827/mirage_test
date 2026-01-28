from run_seg_infer import seg_oct

img_path = "https://storage.googleapis.com/cji102-12/j.png"

image_seg_path = seg_oct(img_path)
print("Segmented image is saved at:", image_seg_path)