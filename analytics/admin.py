from django.contrib import admin
from .models import RealTimeAssetData, ClassificationMetrics

@admin.register(RealTimeAssetData)
class RealTimeAssetDataAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp', 
        'asset_id', 
        'operational_region', 
        'logistics_status', 
        'iot_temperature',
        'traffic_volume_scale',
        'waiting_latency_hours'
    )
    list_filter = (
        'timestamp',
        'asset_id',
        'operational_region',
        'logistics_status'
    )
    search_fields = ('asset_id', 'nlp_classified_genre')
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)
    
    fieldsets = (
        ('Asset Information', {
            'fields': ('asset_id', 'timestamp')
        }),
        ('Location & Status', {
            'fields': ('operational_region', 'logistics_status')
        }),
        ('Real-Time Metrics', {
            'fields': ('iot_temperature', 'traffic_volume_scale', 'waiting_latency_hours')
        }),
        ('Analytics', {
            'fields': ('nlp_classified_genre',)
        }),
    )


@admin.register(ClassificationMetrics)
class ClassificationMetricsAdmin(admin.ModelAdmin):
    list_display = (
        'timestamp',
        'accuracy',
        'precision',
        'recall',
        'f1_score',
        'total_samples',
        'data_source'
    )
    list_filter = (
        'timestamp',
        'data_source',
    )
    search_fields = ('notes',)
    readonly_fields = ('timestamp',)
    ordering = ('-timestamp',)
    
    fieldsets = (
        ('Performance Metrics', {
            'fields': ('accuracy', 'precision', 'recall', 'f1_score', 'total_samples')
        }),
        ('Data Information', {
            'fields': ('timestamp', 'data_source', 'class_labels')
        }),
        ('Matrix & Notes', {
            'fields': ('confusion_matrix_json', 'notes')
        }),
    )
