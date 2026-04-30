from rest_framework import serializers
from django.contrib.auth.models import User
from .models import DetectionHistory


class UserSerializer(serializers.ModelSerializer):
    """Serializer for user information"""
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'date_joined']
        read_only_fields = ['id', 'date_joined']


class DetectionHistorySerializer(serializers.ModelSerializer):
    """Serializer for DetectionHistory model"""
    user = serializers.StringRelatedField(read_only=True)
    formatted_timestamp = serializers.ReadOnlyField()
    detected_classes = serializers.SerializerMethodField()
    
    class Meta:
        model = DetectionHistory
        fields = [
            'id', 
            'user', 
            'timestamp', 
            'formatted_timestamp',
            'detection_data', 
            'image_or_video_ref', 
            'detection_type',
            'confidence_threshold',
            'objects_detected',
            'detected_classes'
        ]
        read_only_fields = ['id', 'timestamp', 'user', 'formatted_timestamp']
    
    def get_detected_classes(self, obj):
        """Get unique detected classes"""
        return obj.get_detected_classes()
    
    def create(self, validated_data):
        """Create DetectionHistory instance with current user"""
        validated_data['user'] = self.context['request'].user
        
        # Count objects detected from detection_data
        detection_data = validated_data.get('detection_data', {})
        if 'detections' in detection_data:
            validated_data['objects_detected'] = len(detection_data['detections'])
        
        return super().create(validated_data)


class DetectionHistoryListSerializer(serializers.ModelSerializer):
    """Simplified serializer for listing detection history"""
    detected_classes = serializers.SerializerMethodField()
    formatted_timestamp = serializers.ReadOnlyField()
    
    class Meta:
        model = DetectionHistory
        fields = [
            'id',
            'timestamp',
            'formatted_timestamp',
            'detection_type',
            'objects_detected',
            'confidence_threshold',
            'detected_classes',
            'image_or_video_ref',
            'detection_data'
        ]
    
    def get_detected_classes(self, obj):
        """Get unique detected classes"""
        return obj.get_detected_classes()


class DetectionHistoryCreateSerializer(serializers.ModelSerializer):
    """Serializer for creating detection history entries"""
    
    class Meta:
        model = DetectionHistory
        fields = [
            'detection_data',
            'image_or_video_ref',
            'detection_type',
            'confidence_threshold'
        ]
    
    def create(self, validated_data):
        """Create DetectionHistory instance with current user"""
        validated_data['user'] = self.context['request'].user
        
        # Count objects detected from detection_data
        detection_data = validated_data.get('detection_data', {})
        if 'detections' in detection_data:
            validated_data['objects_detected'] = len(detection_data['detections'])
        
        return super().create(validated_data)