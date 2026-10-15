from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('objectives/', views.objectives_view, name='objectives'),
    path('chatbot/', views.chatbot_view, name='chatbot'),
    path('chatbot/api/', views.chatbot_api, name='chatbot_api'),
    path('api/latest-metrics/', views.latest_classification_metrics, name='latest_classification_metrics'),
    
    # Real-time Asset Data URLs
    path('realtime-data/', views.realtime_data_view, name='realtime_data'),
    path('realtime-data/add/', views.add_realtime_data, name='add_realtime_data'),
    path('api/realtime-data/add/', views.add_realtime_data_api, name='add_realtime_data_api'),
    path('api/realtime-data/', views.realtime_data_api, name='realtime_data_api'),
]

