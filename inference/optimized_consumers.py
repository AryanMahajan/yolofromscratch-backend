import json
import cv2
import base64
import numpy as np
import logging
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from ultralytics import YOLO
from collections import Counter

# Get an instance of a logger
logger = logging.getLogger(__name__)

class OptimizedVideoConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model once per consumer instance
        self.model = YOLO('runs/detect/train/weights/best.pt')
        # Configure model for speed
        self.model.conf = 0.5  # Lower confidence threshold
        self.model.iou = 0.45  # Lower IoU threshold
        self.model.max_det = 50  # Limit max detections
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Frame processing control
        self.processing = False
        self.last_process_time = 0
        self.min_frame_interval = 1/30  # Max 30 FPS processing
        
        # Frame skip counter for adaptive processing
        self.frame_count = 0
        self.skip_frames = 1  # Process every nth frame initially
        
        # Enhanced detection session tracking
        self.detection_session = {
            'total_frames': 0,
            'total_objects': 0,
            'seen_track_ids': set(),  # Track unique object IDs
            'detections': [],
            'confidence_threshold': 0.5,
            'class_counts': Counter(),  # Track all detected classes
            'all_detections': [],  # Store all individual detections
            'session_start_time': time.time()
        }

    async def connect(self):
        await self.accept()
        logger.info("Optimized WebSocket connected.")
        self.detection_session['session_start_time'] = time.time()

    async def disconnect(self, close_code):
        logger.info(f"Optimized WebSocket disconnected with close code: {close_code}")
        
        # Save detection session to history if user is authenticated
        if hasattr(self, 'user') and self.user and self.user.is_authenticated:
            await self.save_detection_session()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)

    @database_sync_to_async
    def save_detection_session(self):
        """Enhanced save detection session with proper class tracking"""
        try:
            from history.models import DetectionHistory
            
            if self.detection_session['total_frames'] > 0:
                # Calculate session duration
                session_duration = time.time() - self.detection_session['session_start_time']
                
                # Get all unique detected classes
                detected_classes = list(self.detection_session['class_counts'].keys())
                
                # Create comprehensive detection data
                detection_data = {
                    'session_summary': {
                        'total_frames': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'session_duration_seconds': round(session_duration, 2),
                        'session_duration_frames': self.detection_session['total_frames'],
                        'detection_mode': 'optimized',
                        'unique_classes_detected': len(detected_classes),
                        'class_distribution': dict(self.detection_session['class_counts'])
                    },
                    'detected_classes_summary': detected_classes,
                    'class_counts': dict(self.detection_session['class_counts']),
                    'recent_detections': self.detection_session['detections'][-10:],  # Last 10 detection frames
                    'sample_detections': self.detection_session['all_detections'][-20:]  # Last 20 individual detections
                }
                
                DetectionHistory.objects.create(
                    user=self.user,
                    detection_data=detection_data,
                    detection_type='live',
                    confidence_threshold=self.detection_session['confidence_threshold'],
                    objects_detected=self.detection_session['total_objects'],
                    detected_classes=detected_classes  # This should populate the detected_classes field
                )
                logger.info(f"Saved optimized detection session for user {self.user.username} with classes: {detected_classes}")
        except Exception as e:
            logger.error(f"Failed to save optimized detection session: {e}")

    def correct_padding(self, s):
        return s + '=' * (-len(s) % 4)

    def process_frame_sync(self, img_data_str, confidence_threshold):
        """Synchronous frame processing with enhanced detection tracking"""
        try:
            # Decode image
            padded_img_data_str = self.correct_padding(img_data_str)
            img_data = base64.b64decode(padded_img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("cv2.imdecode returned None")
                return None

            # Store original dimensions for coordinate scaling
            original_height, original_width = img.shape[:2]

            # Resize image for faster processing (optional)
            if original_width > 640:
                scale = 640 / original_width
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
                scale = 1.0

            # Run YOLO model with optimized settings using track for persistence
            results = self.model.track(img_resized, persist=True, conf=confidence_threshold, verbose=False)
            
            # Extract detection information with enhanced tracking
            detections = []
            frame_classes = []
            track_ids = []
            
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                # Get track IDs if available
                current_track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i])
                    class_name = self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}"
                    confidence = float(boxes.conf[i])
                    track_id = current_track_ids[i]
                    
                    # Get bounding box coordinates and scale back to original size
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id,
                        'track_id': track_id,
                        'bbox': {
                            'x1': float(xyxy[0] / scale),
                            'y1': float(xyxy[1] / scale),
                            'x2': float(xyxy[2] / scale),
                            'y2': float(xyxy[3] / scale)
                        }
                    }
                    detections.append(detection)
                    frame_classes.append(class_name)
                    track_ids.append(track_id)

            # Draw bounding boxes
            annotated_frame = results[0].plot()

            # Encode with lower quality for speed
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            encoded_frame = base64.b64encode(buffer).decode('utf-8')

            return {
                'encoded_frame': encoded_frame,
                'detections': detections,
                'frame_classes': frame_classes,
                'original_dimensions': {'width': original_width, 'height': original_height}
            }

        except Exception as e:
            logger.exception(f"Error in frame processing: {e}")
            return None

    async def receive(self, text_data):
        # Skip frames if still processing previous frame
        if self.processing:
            return
            
        current_time = time.time()
        
        # Rate limiting - don't process frames too frequently
        if current_time - self.last_process_time < self.min_frame_interval:
            return
            
        # Adaptive frame skipping
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return

        self.processing = True
        self.last_process_time = current_time

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

            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self.process_frame_sync, 
                img_data_str,
                self.detection_session['confidence_threshold']
            )

            if result:
                detections = result['detections']
                frame_classes = result['frame_classes']
                
                # Update session statistics with enhanced tracking
                self.detection_session['total_frames'] += 1
                
                # Update unique object tracking and class counts using track_id
                for detection in detections:
                    track_id = detection.get('track_id')
                    class_name = detection.get('class')
                    
                    if track_id is not None:
                        if track_id not in self.detection_session['seen_track_ids']:
                            self.detection_session['seen_track_ids'].add(track_id)
                            self.detection_session['total_objects'] += 1
                            self.detection_session['class_counts'][class_name] += 1
                    else:
                        # Fallback for untracked objects - count every detection
                        self.detection_session['total_objects'] += 1
                        self.detection_session['class_counts'][class_name] += 1

                # Store recent detections with better structure
                if len(detections) > 0:
                    detection_frame = {
                        'frame_number': self.detection_session['total_frames'],
                        'timestamp': int(current_time * 1000),  # Milliseconds
                        'detections': detections,
                        'objects_count': len(detections),
                        'classes_in_frame': list(set(frame_classes)),  # Unique classes in this frame
                        'frame_classes': frame_classes  # All classes including duplicates
                    }
                    self.detection_session['detections'].append(detection_frame)
                    
                    # Also store individual detections for sampling
                    for detection in detections:
                        self.detection_session['all_detections'].append({
                            'frame_number': self.detection_session['total_frames'],
                            'timestamp': int(current_time * 1000),
                            **detection
                        })
                    
                    # Keep only last 50 detection frames in memory
                    if len(self.detection_session['detections']) > 50:
                        self.detection_session['detections'].pop(0)
                    
                    # Keep only last 200 individual detections in memory
                    if len(self.detection_session['all_detections']) > 200:
                        self.detection_session['all_detections'] = self.detection_session['all_detections'][-200:]

                # Send response with enhanced class information
                await self.send(text_data=json.dumps({
                    'frame': f'data:image/jpeg;base64,{result["encoded_frame"]}',
                    'detections': detections,
                    'frame_stats': {
                        'objects_in_frame': len(detections),
                        'classes_in_frame': list(set(frame_classes)),
                        'total_frames_processed': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'unique_classes_detected': len(self.detection_session['class_counts']),
                        'session_class_counts': dict(self.detection_session['class_counts'])
                    }
                }))
                
                # Adaptive frame skipping based on processing time
                processing_time = time.time() - current_time
                if processing_time > 0.1:  # If processing takes > 100ms
                    self.skip_frames = min(self.skip_frames + 1, 5)
                elif processing_time < 0.05:  # If processing < 50ms
                    self.skip_frames = max(self.skip_frames - 1, 1)

        except Exception as e:
            logger.exception(f"An error occurred in receive: {e}")
        finally:
            self.processing = False


class OptimizedCoordsVideoConsumer(AsyncWebsocketConsumer):
    """Optimized version that only sends coordinates with enhanced history tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model once per consumer instance
        self.model = YOLO('runs/detect/train/weights/best.pt')
        # Configure model for speed
        self.model.conf = 0.5
        self.model.iou = 0.45
        self.model.max_det = 50
        
        # Thread pool for CPU-intensive operations
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Frame processing control
        self.processing = False
        self.last_process_time = 0
        self.min_frame_interval = 1/60  # Max 60 FPS processing
        
        # Frame skip counter for adaptive processing
        self.frame_count = 0
        self.skip_frames = 1
        
        # Enhanced detection session tracking
        self.detection_session = {
            'total_frames': 0,
            'total_objects': 0,
            'seen_track_ids': set(),  # Track unique object IDs
            'detections': [],
            'confidence_threshold': 0.5,
            'class_counts': Counter(),
            'all_detections': [],
            'session_start_time': time.time()
        }

    async def connect(self):
        await self.accept()
        logger.info("Optimized Coords WebSocket connected.")
        self.detection_session['session_start_time'] = time.time()

    async def disconnect(self, close_code):
        logger.info(f"Optimized Coords WebSocket disconnected with close code: {close_code}")
        
        # Save detection session to history if user is authenticated
        if hasattr(self, 'user') and self.user and self.user.is_authenticated:
            await self.save_detection_session()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)

    @database_sync_to_async
    def save_detection_session(self):
        """Save detection session with comprehensive class tracking"""
        try:
            from history.models import DetectionHistory
            
            if self.detection_session['total_frames'] > 0:
                session_duration = time.time() - self.detection_session['session_start_time']
                detected_classes = list(self.detection_session['class_counts'].keys())
                
                detection_data = {
                    'session_summary': {
                        'total_frames': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'session_duration_seconds': round(session_duration, 2),
                        'session_duration_frames': self.detection_session['total_frames'],
                        'detection_mode': 'coordinates_only',
                        'unique_classes_detected': len(detected_classes),
                        'class_distribution': dict(self.detection_session['class_counts'])
                    },
                    'detected_classes_summary': detected_classes,
                    'class_counts': dict(self.detection_session['class_counts']),
                    'recent_detections': self.detection_session['detections'][-10:],
                    'sample_detections': self.detection_session['all_detections'][-20:]
                }
                
                DetectionHistory.objects.create(
                    user=self.user,
                    detection_data=detection_data,
                    detection_type='live',
                    confidence_threshold=self.detection_session['confidence_threshold'],
                    objects_detected=self.detection_session['total_objects'],
                    detected_classes=detected_classes
                )
                logger.info(f"Saved coords detection session for user {self.user.username} with classes: {detected_classes}")
        except Exception as e:
            logger.error(f"Failed to save coords detection session: {e}")

    def correct_padding(self, s):
        return s + '=' * (-len(s) % 4)

    def process_coords_only(self, img_data_str, confidence_threshold):
        """Process frame and return coordinates with enhanced class tracking"""
        try:
            # Decode image
            padded_img_data_str = self.correct_padding(img_data_str)
            img_data = base64.b64decode(padded_img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("cv2.imdecode returned None")
                return None

            # Store original dimensions
            original_height, original_width = img.shape[:2]

            # Resize for faster processing if needed
            if original_width > 640:
                scale = 640 / original_width
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                img_resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                img_resized = img
                scale = 1.0

            # Run YOLO model with tracking - only inference, no plotting
            results = self.model.track(img_resized, persist=True, conf=confidence_threshold, verbose=False)
            
            # Extract detection coordinates and info with class tracking
            detections = []
            frame_classes = []
            
            if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                # Get track IDs if available
                current_track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                
                for i in range(len(boxes)):
                    # Get class name
                    class_id = int(boxes.cls[i])
                    class_name = self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}"
                    confidence = float(boxes.conf[i])
                    track_id = current_track_ids[i]
                    
                    # Get bounding box coordinates and scale back to original size
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id,
                        'track_id': track_id,
                        'bbox': {
                            'x1': float(xyxy[0] / scale),
                            'y1': float(xyxy[1] / scale),
                            'x2': float(xyxy[2] / scale),
                            'y2': float(xyxy[3] / scale)
                        }
                    }
                    detections.append(detection)
                    frame_classes.append(class_name)

            return {
                'detections': detections,
                'frame_classes': frame_classes,
                'original_dimensions': {'width': original_width, 'height': original_height}
            }

        except Exception as e:
            logger.exception(f"Error in coords processing: {e}")
            return None

    async def receive(self, text_data):
        # Skip frames if still processing previous frame
        if self.processing:
            return
            
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_process_time < self.min_frame_interval:
            return
            
        # Process more frames since it's faster
        self.frame_count += 1
        if self.frame_count % self.skip_frames != 0:
            return

        self.processing = True
        self.last_process_time = current_time

        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frame']
            
            # Update confidence threshold if provided
            if 'confidence_threshold' in text_data_json:
                self.detection_session['confidence_threshold'] = text_data_json['confidence_threshold']
            
            # Get user from scope
            if hasattr(self.scope, 'user'):
                self.user = self.scope['user']
            
            img_data_str = frame_data.split(',')[1]

            # Run processing in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self.process_coords_only, 
                img_data_str,
                self.detection_session['confidence_threshold']
            )

            if result:
                detections = result['detections']
                frame_classes = result['frame_classes']
                
                # Update session statistics with enhanced tracking
                self.detection_session['total_frames'] += 1
                
                # Update unique object tracking and class counts
                for detection in detections:
                    track_id = detection.get('track_id')
                    class_name = detection.get('class')
                    
                    if track_id is not None:
                        if track_id not in self.detection_session['seen_track_ids']:
                            self.detection_session['seen_track_ids'].add(track_id)
                            self.detection_session['total_objects'] += 1
                            self.detection_session['class_counts'][class_name] += 1
                    else:
                        # Fallback
                        self.detection_session['total_objects'] += 1
                        self.detection_session['class_counts'][class_name] += 1
                
                # Store recent detections
                if len(detections) > 0:
                    detection_frame = {
                        'frame_number': self.detection_session['total_frames'],
                        'timestamp': int(current_time * 1000),
                        'detections': detections,
                        'objects_count': len(detections),
                        'classes_in_frame': list(set(frame_classes)),
                        'frame_classes': frame_classes
                    }
                    self.detection_session['detections'].append(detection_frame)
                    
                    # Store individual detections
                    for detection in detections:
                        self.detection_session['all_detections'].append({
                            'frame_number': self.detection_session['total_frames'],
                            'timestamp': int(current_time * 1000),
                            **detection
                        })
                    
                    # Memory management
                    if len(self.detection_session['detections']) > 50:
                        self.detection_session['detections'].pop(0)
                    
                    if len(self.detection_session['all_detections']) > 200:
                        self.detection_session['all_detections'] = self.detection_session['all_detections'][-200:]

                # Send only coordinates with comprehensive class info
                await self.send(text_data=json.dumps({
                    'detections': detections,
                    'video_dimensions': result['original_dimensions'],
                    'frame_stats': {
                        'objects_in_frame': len(detections),
                        'classes_in_frame': list(set(frame_classes)),
                        'total_frames_processed': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'unique_classes_detected': len(self.detection_session['class_counts']),
                        'session_class_counts': dict(self.detection_session['class_counts'])
                    },
                    'mode': 'coordinates_only'
                }))
                
                # Adaptive processing
                processing_time = time.time() - current_time
                if processing_time > 0.05:
                    self.skip_frames = min(self.skip_frames + 1, 3)
                elif processing_time < 0.02:
                    self.skip_frames = max(self.skip_frames - 1, 1)

        except Exception as e:
            logger.exception(f"An error occurred in coords receive: {e}")
        finally:
            self.processing = False


class UltraLightVideoConsumer(AsyncWebsocketConsumer):
    """Ultra-lightweight version with enhanced history tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load model with minimal settings
        self.model = YOLO('runs/detect/train/weights/best.pt')
        # Aggressive optimization
        self.model.conf = 0.6
        self.model.iou = 0.5
        self.model.max_det = 20
        
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.processing = False
        self.frame_count = 0
        
        # Enhanced detection session tracking
        self.detection_session = {
            'total_frames': 0,
            'total_objects': 0,
            'detections': [],
            'confidence_threshold': 0.6,
            'class_counts': Counter(),
            'all_detections': [],
            'session_start_time': time.time()
        }
        
    async def connect(self):
        await self.accept()
        logger.info("Ultra-light WebSocket connected.")
        self.detection_session['session_start_time'] = time.time()

    async def disconnect(self, close_code):
        logger.info(f"Ultra-light WebSocket disconnected: {close_code}")
        
        # Save detection session to history if user is authenticated
        if hasattr(self, 'user') and self.user and self.user.is_authenticated:
            await self.save_detection_session()
        
        self.executor.shutdown(wait=False)

    @database_sync_to_async
    def save_detection_session(self):
        """Save detection session with comprehensive tracking"""
        try:
            from history.models import DetectionHistory
            
            if self.detection_session['total_frames'] > 0:
                session_duration = time.time() - self.detection_session['session_start_time']
                detected_classes = list(self.detection_session['class_counts'].keys())
                
                detection_data = {
                    'session_summary': {
                        'total_frames': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'session_duration_seconds': round(session_duration, 2),
                        'session_duration_frames': self.detection_session['total_frames'],
                        'detection_mode': 'ultra_light',
                        'unique_classes_detected': len(detected_classes),
                        'class_distribution': dict(self.detection_session['class_counts'])
                    },
                    'detected_classes_summary': detected_classes,
                    'class_counts': dict(self.detection_session['class_counts']),
                    'recent_detections': self.detection_session['detections'][-5:],
                    'sample_detections': self.detection_session['all_detections'][-10:]
                }
                
                DetectionHistory.objects.create(
                    user=self.user,
                    detection_data=detection_data,
                    detection_type='live',
                    confidence_threshold=self.detection_session['confidence_threshold'],
                    objects_detected=self.detection_session['total_objects'],
                    detected_classes=detected_classes
                )
                logger.info(f"Saved ultra-light detection session for user {self.user.username} with classes: {detected_classes}")
        except Exception as e:
            logger.error(f"Failed to save ultra-light detection session: {e}")

    def ultra_fast_process(self, img_data_str, confidence_threshold):
        """Minimal processing with enhanced class tracking"""
        try:
            # Quick decode
            img_data_str += '=' * (-len(img_data_str) % 4)
            img_data = base64.b64decode(img_data_str)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                return None

            # Aggressive resize for speed
            img = cv2.resize(img, (416, 416))

            # Fast inference
            results = self.model(img, conf=confidence_threshold, verbose=False, stream=True)
            result = next(results)
            
            # Extract detections with class tracking
            detections = []
            frame_classes = []
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    class_name = self.model.names[class_id] if class_id in self.model.names else f"class_{class_id}"
                    confidence = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    detection = {
                        'class': class_name,
                        'confidence': confidence,
                        'class_id': class_id,
                        'bbox': {
                            'x1': float(xyxy[0]),
                            'y1': float(xyxy[1]),
                            'x2': float(xyxy[2]),
                            'y2': float(xyxy[3])
                        }
                    }
                    detections.append(detection)
                    frame_classes.append(class_name)
            
            # Quick annotation
            annotated_frame = result.plot()
            
            # Fast encode with low quality
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 60]
            _, buffer = cv2.imencode('.jpg', annotated_frame, encode_params)
            
            return {
                'encoded_frame': base64.b64encode(buffer).decode('utf-8'),
                'detections': detections,
                'frame_classes': frame_classes
            }

        except Exception as e:
            logger.error(f"Ultra-fast processing error: {e}")
            return None

    async def receive(self, text_data):
        # Process every 3rd frame only
        self.frame_count += 1
        if self.processing or self.frame_count % 3 != 0:
            return

        self.processing = True
        current_time = time.time()

        try:
            text_data_json = json.loads(text_data)
            frame_data = text_data_json['frame']
            
            # Update confidence threshold if provided
            if 'confidence_threshold' in text_data_json:
                self.detection_session['confidence_threshold'] = text_data_json['confidence_threshold']
            
            # Get user from scope
            if hasattr(self.scope, 'user'):
                self.user = self.scope['user']
            
            img_data_str = frame_data.split(',')[1]

            # Ultra-fast processing
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.ultra_fast_process,
                img_data_str,
                self.detection_session['confidence_threshold']
            )

            if result:
                detections = result['detections']
                frame_classes = result['frame_classes']
                
                # Update session statistics
                self.detection_session['total_frames'] += 1
                self.detection_session['total_objects'] += len(detections)
                
                # Update class counts
                for class_name in frame_classes:
                    self.detection_session['class_counts'][class_name] += 1
                
                # Store recent detections (minimal storage)
                if len(detections) > 0:
                    detection_frame = {
                        'frame_number': self.detection_session['total_frames'],
                        'timestamp': int(current_time * 1000),
                        'detections': detections,
                        'objects_count': len(detections),
                        'classes_in_frame': list(set(frame_classes)),
                        'frame_classes': frame_classes
                    }
                    self.detection_session['detections'].append(detection_frame)
                    
                    # Store individual detections
                    for detection in detections:
                        self.detection_session['all_detections'].append({
                            'frame_number': self.detection_session['total_frames'],
                            'timestamp': int(current_time * 1000),
                            **detection
                        })
                    
                    # Keep only last 20 detection frames in memory
                    if len(self.detection_session['detections']) > 20:
                        self.detection_session['detections'].pop(0)
                    
                    if len(self.detection_session['all_detections']) > 50:
                        self.detection_session['all_detections'] = self.detection_session['all_detections'][-50:]

                await self.send(text_data=json.dumps({
                    'frame': f'data:image/jpeg;base64,{result["encoded_frame"]}',
                    'detections': detections,
                    'frame_stats': {
                        'objects_in_frame': len(detections),
                        'classes_in_frame': list(set(frame_classes)),
                        'total_frames_processed': self.detection_session['total_frames'],
                        'total_objects_detected': self.detection_session['total_objects'],
                        'confidence_threshold': self.detection_session['confidence_threshold'],
                        'unique_classes_detected': len(self.detection_session['class_counts']),
                        'session_class_counts': dict(self.detection_session['class_counts'])
                    }
                }))

        except Exception as e:
            logger.exception(f"Ultra-light receive error: {e}")
        finally:
            self.processing = False