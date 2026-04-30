import os
import cv2
import tempfile
import uuid
import logging
import subprocess
from django.conf import settings
from rest_framework import views, status, permissions
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from ultralytics import YOLO
from history.models import DetectionHistory
from .serializers import VideoUploadSerializer
from collections import Counter

logger = logging.getLogger(__name__)

class VideoUploadView(views.APIView):
    parser_classes = (MultiPartParser, FormParser)
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = VideoUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        video_file = serializer.validated_data['video']
        conf_threshold = serializer.validated_data.get('confidence_threshold', 0.5)

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video_path = temp_video.name

        try:
            # Load model
            model = YOLO('runs/detect/train/weights/best.pt')
            
            # Open the video
            cap = cv2.VideoCapture(temp_video_path)
            if not cap.isOpened():
                return Response({'error': 'Could not open video file'}, status=status.HTTP_400_BAD_REQUEST)

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Prepare temporary output video (raw OpenCV output)
            raw_output_filename = f"raw_{uuid.uuid4()}.mp4"
            raw_output_path = os.path.join(tempfile.gettempdir(), raw_output_filename)
            
            # Use mp4v for raw output (guaranteed to work with OpenCV)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(raw_output_path, fourcc, fps, (width, height))

            # Detection stats
            seen_track_ids = set()
            class_counts = Counter()
            all_detections = []
            frame_count = 0

            # Process frame by frame
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Use track instead of predict to maintain object identity across frames
                results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
                
                if len(results) > 0:
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                        boxes = results[0].boxes
                        # Extract track IDs if available
                        track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                        
                        for i in range(len(boxes)):
                            class_id = int(boxes.cls[i])
                            class_name = model.names[class_id] if class_id in model.names else f"class_{class_id}"
                            confidence = float(boxes.conf[i])
                            track_id = track_ids[i]
                            
                            # Update unique object tracking
                            if track_id is not None:
                                if track_id not in seen_track_ids:
                                    seen_track_ids.add(track_id)
                                    class_counts[class_name] += 1
                            else:
                                # Fallback if tracking is not working: count as new object
                                # (But usually track() with persist=True works well)
                                class_counts[class_name] += 1
                            
                            if frame_count % 10 == 0: # Sample detections
                                xyxy = boxes.xyxy[i].cpu().numpy()
                                all_detections.append({
                                    'frame': frame_count,
                                    'class': class_name,
                                    'confidence': confidence,
                                    'track_id': track_id,
                                    'bbox': [float(x) for x in xyxy]
                                })
                else:
                    out.write(frame)

            cap.release()
            out.release()
            
            total_objects = len(seen_track_ids) if seen_track_ids else sum(class_counts.values())

            # Transcode to H.264 using FFmpeg for browser compatibility
            final_filename = f"processed_{uuid.uuid4()}.mp4"
            output_dir = os.path.join(settings.MEDIA_ROOT, 'processed_videos')
            os.makedirs(output_dir, exist_ok=True)
            final_output_path = os.path.join(output_dir, final_filename)
            
            ffmpeg_command = [
                'ffmpeg', '-y', '-i', raw_output_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'fast', '-crf', '23',
                final_output_path
            ]
            
            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                # Cleanup raw output
                if os.path.exists(raw_output_path):
                    os.remove(raw_output_path)
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg error: {e.stderr.decode()}")
                # If FFmpeg fails, just move the raw output as a fallback
                os.rename(raw_output_path, final_output_path)

            # Create DetectionHistory entry
            detected_classes = list(class_counts.keys())
            detection_data = {
                'summary': {
                    'total_frames': frame_count,
                    'total_objects_detected': total_objects,
                    'class_distribution': dict(class_counts),
                    'fps': fps,
                    'duration': round(frame_count / fps, 2) if fps > 0 else 0
                },
                'detected_classes': detected_classes,
                'sample_detections': all_detections[:100]
            }

            video_url = f"{settings.MEDIA_URL}processed_videos/{final_filename}"
            
            history = DetectionHistory.objects.create(
                user=request.user,
                detection_data=detection_data,
                detection_type='video',
                image_or_video_ref=video_url,
                confidence_threshold=conf_threshold,
                objects_detected=total_objects
            )

            return Response({
                'id': history.id,
                'video_url': video_url,
                'stats': detection_data['summary'],
                'detected_classes': detected_classes
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.exception(f"Error processing video: {e}")
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if 'raw_output_path' in locals() and os.path.exists(raw_output_path):
                os.remove(raw_output_path)
