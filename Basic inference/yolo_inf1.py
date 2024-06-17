"""Doing inference with yolov8 model."""

from PIL import Image
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# Detect from webcam
# results = model.predict(source="0", show=True)

# from PIL
im1 = Image.open("traffic.jpg")
results = model.predict(source=im1, save=True)  # save plotted images

# from mp4
results = model.predict("farm.mp4", save=True)
