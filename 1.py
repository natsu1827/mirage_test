from run_cls_infer import cls_oct

img_path = "https://storage.googleapis.com/cji102-12/j.png"

pred, conf = cls_oct(img_path)
print("Prediction:", pred)
print("Confidence:", conf)