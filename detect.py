import sys
import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # Force Windows path handling on Windows

from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import torch
import numpy as np
import random
from ultralytics import YOLO


def convertCVImage2QtImage(cv_img):
    """Convert a cv2 image to a Qt QPixmap."""
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    qimg = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class Detector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        weights_path = "C:\\Y4\\Y41\\project\\yolov5\\runnew\\weight\\last1.pt"  # Update with your path
        try:
            self.model = YOLO(weights_path)  # Load YOLO model
            self.model.to(self.device).eval()
            print("Available classes in the model:", self.model.names)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            sys.exit(1)

        self.class_names = self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.class_names))]

    def detect(self, frame):
        results = self.model(frame)  # Perform inference
        object_counts = {class_name: 0 for class_name in self.class_names}

        for result in results[0].boxes.data:  # Iterate through detections
            x1, y1, x2, y2, conf, cls = result.tolist()
            class_name = self.class_names[int(cls)]
            label = f"{class_name} {conf:.2f}"
            color = self.colors[int(cls)]

            # Convert bounding box coordinates to integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Increment object count safely
            object_counts[class_name] = object_counts.get(class_name, 0) + 1

        return frame, object_counts


class ProcessImage(QThread):
    signal_show_frame = Signal(object)
    signal_object_counts = Signal(dict)

    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.detector = Detector()
        self.running = True

    def run(self):
        self.video = cv2.VideoCapture(self.source)
        if not self.video.isOpened():
            print("Error: Could not open webcam.")
            return

        while self.running:
            valid, frame = self.video.read()
            if not valid:
                break

            # Resize frame to reduce processing load
            frame = cv2.resize(frame, (640, 480))

            # Perform detection
            frame, object_counts = self.detector.detect(frame)

            # Emit signals
            self.signal_show_frame.emit(frame)
            self.signal_object_counts.emit(object_counts)

            cv2.waitKey(30)

        self.video.release()

    def stop(self):
        self.running = False
        self.wait()  # Wait for thread to finish


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection")
        self.setGeometry(0, 0, 1920, 1080)  # Fullscreen window
        self.layout = QVBoxLayout(self)

        # Camera feed label
        self.lbl_camera = QLabel("Loading camera...", self)
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setFixedSize(1920, 1080)
        self.layout.addWidget(self.lbl_camera)

        # Start the image processing thread
        self.process_thread = ProcessImage(0)
        self.process_thread.signal_show_frame.connect(self.show_frame)
        self.process_thread.signal_object_counts.connect(self.update_object_counts)
        self.process_thread.start()

    def show_frame(self, frame):
        # Adjust frame size to match the label's size
        height, width = self.lbl_camera.height(), self.lbl_camera.width()
        frame_resized = cv2.resize(frame, (width, height))
        pixmap = convertCVImage2QtImage(frame_resized)
        self.lbl_camera.setPixmap(pixmap)

    def update_object_counts(self, object_counts):
        # Display object counts in the console (can be updated to show on GUI)
        print("Detected objects:", object_counts)

    def closeEvent(self, event):
        self.process_thread.stop()  # Stop the thread
        event.accept()  # Close the application


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
