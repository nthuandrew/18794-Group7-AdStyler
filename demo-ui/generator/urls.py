from django.urls import path
from . import views

app_name = 'generator'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/generate/', views.generate_image, name=''),
    path('api/history/', views.history, name='history'),
]

