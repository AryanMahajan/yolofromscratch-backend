from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from accounts.views import CustomConfirmEmailView
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

schema_view = get_schema_view(
    openapi.Info(
        title="YOLO API",
        default_version='v1',
        description="API for YOLO object detection with JWT authentication",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@yolo.local"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    path('admin/', admin.site.urls),

    # 1️⃣ Custom email confirm view must come BEFORE the default include
    path(
        "api/auth/registration/account-confirm-email/<str:key>/",
        CustomConfirmEmailView.as_view(),
        name="account_confirm_email",
    ),

    # 2️⃣ JWT and Account management routes
    path('accounts/', include('accounts.urls')),  # JWT tokens, user/me, Google login
    path('api/auth/', include('dj_rest_auth.urls')),
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),

    # 3️⃣ Detection History API
    path('', include('history.urls')),

    # 4️⃣ Inference API (Video Upload, etc.)
    path('api/inference/', include('inference.urls')),

    # 5️⃣ API Documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)