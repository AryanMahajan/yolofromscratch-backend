from rest_framework import serializers

class VideoUploadSerializer(serializers.Serializer):
    video = serializers.FileField()
    confidence_threshold = serializers.FloatField(default=0.5, required=False)
