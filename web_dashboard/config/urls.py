"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from dashboard import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('start_session', views.start_session, name='start_session'),
    path('stop_session', views.stop_session, name='stop_session'),
    path('control_session', views.control_session, name='control_session'),
    path('get_stats', views.get_stats, name='get_stats'),
    # MP4 Processing
    path('process-mp4', views.mp4_page, name='mp4_page'),
    path('upload-mp4', views.upload_mp4, name='upload_mp4'),
    path('start-mp4', views.start_mp4, name='start_mp4'),
    path('stop-mp4', views.stop_mp4, name='stop_mp4'),
    path('mp4-progress', views.mp4_progress, name='mp4_progress'),
    path('mp4-results', views.mp4_results, name='mp4_results'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
