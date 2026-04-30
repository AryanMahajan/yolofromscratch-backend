from django.urls import path
from .views import VideoUploadView

urlpatterns = [
    path('upload-video/', VideoUploadView.as_view(), name='upload-video'),
]
