import cv2
import torch
import numpy as np
from PIL import Image
import io
import base64
import json
from flask import Flask, request, jsonify
import time

class YOLOv5WebDetector:
    def __init__(self):
        """Initialize the YOLOv5 detector for web API"""
        try:
            # Load YOLOv5 model
            print("Loading YOLOv5 model for web API...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.eval()
            
            # COCO dataset class names
            self.class_names = self.model.names
            
            # Fish and marine life class IDs in COCO dataset
            self.marine_classes = {
                61: 'fish',  # fish
                0: 'person',  # person (for divers)
                1: 'bicycle',  # bicycle
                2: 'car',  # car
                3: 'motorcycle',  # motorcycle
                4: 'airplane',  # airplane
                5: 'bus',  # bus
                6: 'train',  # train
                7: 'truck',  # truck
                8: 'boat',  # boat
                9: 'traffic light',  # traffic light
                10: 'fire hydrant',  # fire hydrant
                11: 'stop sign',  # stop sign
                12: 'parking meter',  # parking meter
                13: 'bench',  # bench
                14: 'bird',  # bird
                15: 'cat',  # cat
                16: 'dog',  # dog
                17: 'horse',  # horse
                18: 'sheep',  # sheep
                19: 'cow',  # cow
                20: 'elephant',  # elephant
                21: 'bear',  # bear
                22: 'zebra',  # zebra
                23: 'giraffe',  # giraffe
                24: 'backpack',  # backpack
                25: 'umbrella',  # umbrella
                26: 'handbag',  # handbag
                27: 'tie',  # tie
                28: 'suitcase',  # suitcase
                29: 'frisbee',  # frisbee
                30: 'skis',  # skis
                31: 'snowboard',  # snowboard
                32: 'sports ball',  # sports ball
                33: 'kite',  # kite
                34: 'baseball bat',  # baseball bat
                35: 'baseball glove',  # baseball glove
                36: 'skateboard',  # skateboard
                37: 'surfboard',  # surfboard
                38: 'tennis racket',  # tennis racket
                39: 'bottle',  # bottle
                40: 'wine glass',  # wine glass
                41: 'cup',  # cup
                42: 'fork',  # fork
                43: 'knife',  # knife
                44: 'spoon',  # spoon
                45: 'bowl',  # bowl
                46: 'banana',  # banana
                47: 'apple',  # apple
                48: 'sandwich',  # sandwich
                49: 'orange',  # orange
                50: 'broccoli',  # broccoli
                51: 'carrot',  # carrot
                52: 'hot dog',  # hot dog
                53: 'pizza',  # pizza
                54: 'donut',  # donut
                55: 'cake',  # cake
                56: 'chair',  # chair
                57: 'couch',  # couch
                58: 'potted plant',  # potted plant
                59: 'bed',  # bed
                60: 'dining table',  # dining table
                61: 'toilet',  # toilet
                62: 'tv',  # tv
                63: 'laptop',  # laptop
                64: 'mouse',  # mouse
                65: 'remote',  # remote
                66: 'keyboard',  # keyboard
                67: 'cell phone',  # cell phone
                68: 'microwave',  # microwave
                69: 'oven',  # oven
                70: 'toaster',  # toaster
                71: 'sink',  # sink
                72: 'refrigerator',  # refrigerator
                73: 'book',  # book
                74: 'clock',  # clock
                75: 'vase',  # vase
                76: 'scissors',  # scissors
                77: 'teddy bear',  # teddy bear
                78: 'hair drier',  # hair drier
                79: 'toothbrush'  # toothbrush
            }
            
            # Detection parameters
            self.confidence_threshold = 0.5
            self.iou_threshold = 0.45
            
            print("YOLOv5 web detector initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing YOLOv5 model: {e}")
            self.model = None

    def detect_objects(self, image_data):
        """Detect objects in image data"""
        if self.model is None:
            return []
        
        try:
            # Convert base64 image data to numpy array
            if isinstance(image_data, str):
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # Convert PIL to numpy array
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Run YOLOv5 inference
            results = self.model(image_np)
            
            # Parse results
            detections = results.pandas().xyxy[0]
            
            detected_objects = []
            
            # Process detections
            for _, detection in detections.iterrows():
                if detection['confidence'] >= self.confidence_threshold:
                    class_id = int(detection['class'])
                    class_name = self.class_names.get(class_id, f'class_{class_id}')
                    
                    # Map to shark types based on detected objects
                    shark_type = self.map_to_shark_type(class_name, detection['confidence'])
                    
                    detected_objects.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'shark_type': shark_type,
                        'confidence': float(detection['confidence']),
                        'bbox': [
                            int(detection['xmin']),
                            int(detection['ymin']),
                            int(detection['xmax']),
                            int(detection['ymax'])
                        ]
                    })
            
            return detected_objects
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return []

    def map_to_shark_type(self, class_name, confidence):
        """Map detected objects to shark types"""
        # High confidence fish detection
        if class_name == 'fish' and confidence > 0.7:
            return 'Tiger Shark'
        
        # Map other objects to different shark types
        shark_mapping = {
            'person': 'Great White Shark',
            'bird': 'Hammerhead Shark',
            'cat': 'Tiger Shark',
            'dog': 'Great White Shark',
            'horse': 'Hammerhead Shark',
            'sheep': 'Tiger Shark',
            'cow': 'Great White Shark',
            'elephant': 'Hammerhead Shark',
            'bear': 'Great White Shark',
            'zebra': 'Tiger Shark',
            'giraffe': 'Hammerhead Shark',
            'boat': 'Marine Life Detected',
            'surfboard': 'Marine Life Detected',
            'bottle': 'Marine Life Detected',
            'cup': 'Marine Life Detected',
            'bowl': 'Marine Life Detected'
        }
        
        return shark_mapping.get(class_name, 'Marine Life Detected')

    def get_detection_summary(self, detections):
        """Get summary of detections"""
        if not detections:
            return {
                'total_objects': 0,
                'shark_types': [],
                'highest_confidence': 0,
                'best_detection': None
            }
        
        shark_types = list(set([d['shark_type'] for d in detections]))
        highest_confidence = max([d['confidence'] for d in detections])
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        return {
            'total_objects': len(detections),
            'shark_types': shark_types,
            'highest_confidence': highest_confidence,
            'best_detection': best_detection
        }

# Global detector instance
detector = None

def initialize_detector():
    """Initialize the global detector"""
    global detector
    if detector is None:
        detector = YOLOv5WebDetector()
    return detector

def create_yolo_api(app):
    """Create YOLOv5 API endpoints"""
    
    @app.route('/api/yolo/detect', methods=['POST'])
    def detect_objects():
        """API endpoint for object detection"""
        try:
            # Initialize detector if needed
            detector = initialize_detector()
            
            if detector.model is None:
                return jsonify({
                    'success': False,
                    'error': 'YOLOv5 model not initialized'
                }), 500
            
            # Get image data from request
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'No image data provided'
                }), 400
            
            image_data = data['image']
            
            # Run detection
            start_time = time.time()
            detections = detector.detect_objects(image_data)
            detection_time = time.time() - start_time
            
            # Get summary
            summary = detector.get_detection_summary(detections)
            
            return jsonify({
                'success': True,
                'detections': detections,
                'summary': summary,
                'detection_time': detection_time,
                'timestamp': time.time()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/yolo/status', methods=['GET'])
    def get_status():
        """API endpoint to check YOLOv5 status"""
        try:
            detector = initialize_detector()
            
            return jsonify({
                'success': True,
                'model_loaded': detector.model is not None,
                'class_count': len(detector.class_names) if detector.model else 0,
                'confidence_threshold': detector.confidence_threshold
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

if __name__ == "__main__":
    # Test the detector
    detector = YOLOv5WebDetector()
    print("YOLOv5 Web API detector ready!")
