from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s_playing_cards.pt")  # load a pretrained model (recommended for training)

# Predict on an image

results = model("/Users/maxspier/Documents/GitHub/Playing-Cards-Detection-with-YoloV8")

print("START")
print(results)
