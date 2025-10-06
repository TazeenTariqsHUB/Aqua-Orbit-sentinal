import cv2
import torch
import numpy as np
from PIL import Image
import requests
import io
import sys
import time


class FishDetector:
    def __init__(self, use_gui=True):
        """Initialize the fish detector with YOLOv5 model"""
        self.use_gui = use_gui

        try:
            # Load YOLOv5 model (this will download it automatically on first run)
            print("Loading YOLOv5 model... (this may take a while on first run)")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()

            # COCO dataset class names - fish is class 61
            self.fish_class_id = 61  # 'fish' in COCO dataset

            # Detection parameters
            self.confidence_threshold = 0.5
            self.iou_threshold = 0.45

            print("Fish detector initialized successfully!")

        except Exception as e:
            print(f"Error initializing model: {e}")
            self.model = None

    def detect_fish(self, frame):
        """Detect fish in the given frame"""
        if self.model is None:
            return frame, []

        try:
            # Convert BGR to RGB for YOLO
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run inference
            results = self.model(rgb_frame)

            # Parse results
            detections = results.pandas().xyxy[0]  # Get detections as pandas DataFrame

            fish_detections = []

            # Filter for fish detections
            for _, detection in detections.iterrows():
                if detection['class'] == self.fish_class_id and detection['confidence'] >= self.confidence_threshold:
                    fish_detections.append({
                        'bbox': [int(detection['xmin']), int(detection['ymin']),
                                 int(detection['xmax']), int(detection['ymax'])],
                        'confidence': detection['confidence']
                    })

            return frame, fish_detections

        except Exception as e:
            print(f"Error during detection: {e}")
            return frame, []

    def draw_bounding_boxes(self, frame, detections):
        """Draw bounding boxes around detected fish"""
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']

            # Extract coordinates
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Create label with confidence
            label = f"Fish: {confidence:.2f}"

            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def save_frame_with_detections(self, frame, detections, frame_count):
        """Save frame with detection results as image file"""
        # Draw bounding boxes on a copy of the frame
        frame_with_boxes = self.draw_bounding_boxes(frame.copy(), detections)

        # Save the frame
        filename = f"fish_detection_frame_{frame_count}_{len(detections)}_fish.jpg"
        cv2.imwrite(filename, frame_with_boxes)
        print(f"Saved frame with {len(detections)} fish detections: {filename}")

        return filename


def check_opencv_gui_support():
    """Check if OpenCV has GUI support"""
    try:
        # Try to create a test window
        cv2.namedWindow('test_window', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test_window')
        return True
    except cv2.error as e:
        if "The function is not implemented" in str(e):
            return False
        return True
    except Exception:
        return False


def run_without_gui(detector):
    """Run fish detection without GUI (headless mode)"""
    print("Running in headless mode (no GUI display)")
    print("Images will be saved automatically when fish are detected")
    print("Press Ctrl+C to stop")

    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0
    last_save_time = 0
    save_interval = 3  # Save every 3 seconds when fish detected

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from camera")
                break

            # Detect fish
            frame, fish_detections = detector.detect_fish(frame)

            current_time = time.time()

            if fish_detections:
                print(f"Frame {frame_count}: Detected {len(fish_detections)} fish")

                # Save frame if enough time has passed
                if current_time - last_save_time >= save_interval:
                    detector.save_frame_with_detections(frame, fish_detections, frame_count)
                    last_save_time = current_time

            # Print status every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")

            frame_count += 1

            # Small delay to prevent excessive CPU usage
            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        print("Camera released")


def run_with_gui(detector):
    """Run fish detection with GUI display"""
    print("Running with GUI display")
    print("Press 'q' to quit, 's' to save current frame")

    # Initialize camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from camera")
                break

            # Detect fish
            frame, fish_detections = detector.detect_fish(frame)

            # Draw bounding boxes
            if fish_detections:
                frame = detector.draw_bounding_boxes(frame, fish_detections)
                print(f"Frame {frame_count}: Detected {len(fish_detections)} fish")

            # Add frame info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            # Display frame
            cv2.imshow('Fish Detection', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                detector.save_frame_with_detections(frame, fish_detections, frame_count)

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")


def main():
    """Main function to run the fish detection"""
    print("Initializing Fish Detection System...")

    # Check if OpenCV has GUI support
    has_gui = check_opencv_gui_support()

    if not has_gui:
        print("WARNING: OpenCV GUI support not available")
        print("This usually happens with headless OpenCV installations")
        print("The program will run in headless mode and save images when fish are detected")

    # Initialize detector
    detector = FishDetector(use_gui=has_gui)

    if detector.model is None:
        print("Failed to initialize model. Exiting...")
        return

    print("Starting fish detection...")

    # Run appropriate mode based on GUI availability
    if has_gui:
        run_with_gui(detector)
    else:
        run_without_gui(detector)


if __name__ == "__main__":
    main()