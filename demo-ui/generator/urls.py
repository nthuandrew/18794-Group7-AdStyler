from django.urls import path
from . import views

app_name = 'generator'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/generate/', views.generate_image, name='generate'),
    path('api/record/<int:record_id>/', views.get_record_status, name='record_status'),
    path('api/history/', views.history, name='history'),
]

