from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class DetectionHistory(models.Model):
    """
    Model to store user's detection history
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='detection_history')
    timestamp = models.DateTimeField(default=timezone.now, db_index=True)
    detection_data = models.JSONField(help_text="JSON data containing detection results")
    image_or_video_ref = models.CharField(
        max_length=500, 
        blank=True, 
        null=True,
        help_text="Reference to stored image/video file or base64 data"
    )
    detection_type = models.CharField(
        max_length=20,
        choices=[
            ('image', 'Image Detection'),
            ('video', 'Video Detection'),
            ('live', 'Live Feed Detection')
        ],
        default='live'
    )
    confidence_threshold = models.FloatField(default=0.5)
    objects_detected = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Detection History'
        verbose_name_plural = 'Detection Histories'
    
    def __str__(self):
        return f"{self.user.username} - {self.detection_type} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"
    
    @property
    def formatted_timestamp(self):
        return self.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def get_detected_classes(self):
        """Extract unique detected classes from detection_data"""
        if not self.detection_data:
            return []

        # Check if already present as a list (e.g. from VideoUploadView)
        if 'detected_classes' in self.detection_data:
            return self.detection_data['detected_classes']

        # Extract from detections list (e.g. from live detection)
        if 'detections' in self.detection_data:
            classes = set()
            for detection in self.detection_data.get('detections', []):
                if 'class' in detection:
                    classes.add(detection['class'])
            return list(classes)

        return []