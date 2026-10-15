from django import forms
from .models import RealTimeAssetData

class ChatbotForm(forms.Form):
    question = forms.CharField(
        label='Ask the AI Chatbot (English / Hindi / Urdu)',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Ask anything: greeting, daily question, math, or supply chain analytics...'
        })
    )


class RealTimeAssetDataForm(forms.ModelForm):
    """Form for entering real-time asset data"""
    class Meta:
        model = RealTimeAssetData
        fields = [
            'asset_id',
            'operational_region',
            'logistics_status',
            'iot_temperature',
            'traffic_volume_scale',
            'waiting_latency_hours',
            'nlp_classified_genre'
        ]
        widgets = {
            'asset_id': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Truck 1, Truck 2',
                'required': True
            }),
            'operational_region': forms.Select(attrs={
                'class': 'form-control',
            }),
            'logistics_status': forms.Select(attrs={
                'class': 'form-control',
            }),
            'iot_temperature': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Temperature in °C',
                'step': '0.1'
            }),
            'traffic_volume_scale': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Volume scale (0-100)',
                'step': '0.1'
            }),
            'waiting_latency_hours': forms.NumberInput(attrs={
                'class': 'form-control',
                'placeholder': 'Hours',
                'step': '0.1'
            }),
            'nlp_classified_genre': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'e.g., Perishable, Fragile, etc.'
            }),
        }
