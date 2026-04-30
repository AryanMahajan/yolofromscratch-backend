import json
import cv2
import base64
import numpy as np
import logging
from collections import Counter
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from ultralytics import YOLO
from django.contrib.auth.models import User

# Get an instance of a logger
logger = logging.getLogger(__name__)

# Load the YOLO model
model = YOLO('runs/detect/train/weights/best.pt')

def correct_padding(s):
    return s + '=' * (-len(s) % 4)

class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Accept the connection
        await self.accept()
        logger.info("WebSocket connected.")
        
        # Initialize session data
        self.detection_session = {
            'total_frames': 0,
            'total_objects': 0,
            'seen_track_ids': set(),
            'class_counts': Counter(),
            'detections': [],
            'confidence_threshold': 0.5
        }

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected with close code: {close_code}")
        
        # Save detection session to history if user is authenticated
        if hasattr(self, 'user') and self.user and self.user.is_authenticated:
            await self.save_detection_session()

    @database_sync_to_async
    def save_detection_session(self):
        """Save the detection session to history"""
        try:
            from history.models import DetectionHistory
            
            if self.detection_session['total_frames'] > 0:
                detected_classes = list(self.detection_session['class_counts'].keys())
                DetectionHistory.objects.create(
                    user=self.user,
                    detection_data={
                        'session_summary': {
                            'total_frames': self.detection_session['total_frames'],
                            'total_objects_detected': self.detection_session['total_objects'],
                            'confidence_threshold': self.detection_session['confidence_threshold'],
                            'session_duration_frames': self.detection_session['total_frames'],
                            'class_counts': dict(self.detection_session['class_counts'])
                        },
                        'detected_classes': detected_classes,
                        'detections': self.detection_session['detections'][-10:]  # Keep last 10 detections
                    },
                    detection_type='live',
                    confidence_threshold=self.detection_session['confidence_threshold'],
                    objects_detected=self.detection_session['total_objects'],
                    detected_classes=detected_classes
                )
                logger.info(f"Saved detection session for user {self.user.username}")
        except Exception as e:
            logger.error(f"Failed to save detection session: {e}")

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frame']
            
            # Update confidence threshold if provided
            if 'confidence_threshold' in text_data_json:
                self.detection_session['confidence_threshold'] = text_data_json['confidence_threshold']
            
            # Get user from scope (if authenticated via middleware)
            if hasattr(self.scope, 'user'):
                self.user = self.scope['user']

            img_data_str = frame_data.split(',')[1]
            padded_img_data_str = correct_padding(img_data_str)
            img_data = base64.b64decode(padded_img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("cv2.imdecode returned None. The image data is likely corrupted or in an unsupported format.")
                return

            # Run YOLO model with confidence threshold and tracking
            conf_threshold = self.detection_session['confidence_threshold']
            results = model.track(img, persist=True, conf=conf_threshold, verbose=False)

            # Extract detection information
            detections = []
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                # Get track IDs if available
                track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i])
                    class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                    confidence = float(boxes.conf[i])
                    track_id = track_ids[i]
                    
                    # Get bounding box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'track_id': track_id,
                        'bbox': {
                            'x1': float(xyxy[0]),
                            'y1': float(xyxy[1]),
                            'x2': float(xyxy[2]),
                            'y2': float(xyxy[3])
                        }
                    }
                    detections.append(detection)

                    # Update session statistics only for new track_ids
                    if track_id is not None:
                        if track_id not in self.detection_session['seen_track_ids']:
                            self.detection_session['seen_track_ids'].add(track_id)
                            self.detection_session['total_objects'] += 1
                            self.detection_session['class_counts'][class_name] += 1
                    else:
                        # Fallback for untracked objects
                        self.detection_session['total_objects'] += 1
                        self.detection_session['class_counts'][class_name] += 1

            # Update session statistics
            self.detection_session['total_frames'] += 1
            
            # Store recent detections (keep last 50 for memory efficiency)
            if len(detections) > 0:
                detection_frame = {
                    'frame_number': self.detection_session['total_frames'],
                    'timestamp': None,  # Will be set when saved to database
                    'detections': detections,
                    'objects_count': len(detections)
                }
                self.detection_session['detections'].append(detection_frame)
                
                # Keep only last 50 detection frames in memory
                if len(self.detection_session['detections']) > 50:
                    self.detection_session['detections'].pop(0)

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Encode the frame back to base64
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            # Send the processed frame back with detection info
            await self.send(text_data=json.dumps({
                'frame': 'data:image/jpeg;base64,' + encoded_frame,
                'detections': detections,
                'frame_stats': {
                    'objects_in_frame': len(detections),
                    'total_frames_processed': self.detection_session['total_frames'],
                    'total_objects_detected': self.detection_session['total_objects'],
                    'confidence_threshold': conf_threshold,
                    'class_counts': dict(self.detection_session['class_counts'])
                }
            }))
            
            logger.debug(f"Processed frame {self.detection_session['total_frames']} with {len(detections)} detections")

        except Exception as e:
            logger.exception(f"An error occurred in receive: {e}")
            await self.send(text_data=json.dumps({
                'error': 'Processing failed',
                'message': str(e)
            }))