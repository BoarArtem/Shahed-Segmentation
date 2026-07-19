from ultralytics import YOLO
import cv2

model = YOLO("runs/segment/train/weights/best.pt")

def photo_inference(img_path, file_name):
    result = model(img_path)

    annotated_frame = result[0].plot(boxes=False)

    cv2.imwrite(file_name, annotated_frame)

def video_inference(video_path):
    return model.predict(
        source=video_path,
        save=True,
        conf=0.2
    )


