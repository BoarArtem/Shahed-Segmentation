from ultralytics import YOLO

class ShahedSegmentation:
    def __init__(self):
        self.model = YOLO("yolo11m-seg")
        self.data_path = "data/data.yaml"
        self.epochs = 50

    def train(self):
        return self.model.train(
            data=self.data_path,
            epochs=self.epochs
        )


model = ShahedSegmentation()

if __name__ == "__main__":
    model.train()