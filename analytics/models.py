from django.db import models

class RealTimeAssetData(models.Model):
    """
    Real-time data warehouse for supply chain assets (trucks, shipments, etc.)
    """
    LOGISTICS_STATUS_CHOICES = [
        ('Advance shipping', 'Advance shipping'),
        ('In Transit', 'In Transit'),
        ('Delayed', 'Delayed'),
        ('Delivered', 'Delivered'),
        ('Pending', 'Pending'),
    ]
    
    REGION_CHOICES = [
        ('Southeast Asia', 'Southeast Asia'),
        ('East Asia', 'East Asia'),
        ('South Asia', 'South Asia'),
        ('Europe', 'Europe'),
        ('North America', 'North America'),
        ('South America', 'South America'),
        ('Africa', 'Africa'),
        ('Middle East', 'Middle East'),
    ]
    
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    asset_id = models.CharField(
        max_length=100, 
        help_text="Asset identifier (e.g., Truck 1, Truck 2, etc.)"
    )
    operational_region = models.CharField(
        max_length=100,
        choices=REGION_CHOICES,
        default='Southeast Asia'
    )
    logistics_status = models.CharField(
        max_length=50,
        choices=LOGISTICS_STATUS_CHOICES,
        default='In Transit'
    )
    iot_temperature = models.FloatField(
        null=True,
        blank=True,
        help_text="Temperature in Celsius"
    )
    traffic_volume_scale = models.FloatField(
        null=True,
        blank=True,
        help_text="Traffic volume on a scale (0-100)"
    )
    waiting_latency_hours = models.FloatField(
        null=True,
        blank=True,
        help_text="Waiting latency in hours"
    )
    nlp_classified_genre = models.CharField(
        max_length=100,
        null=True,
        blank=True,
        help_text="NLP classified genre/category"
    )
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Real Time Asset Data'
        verbose_name_plural = 'Real Time Asset Data'
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['asset_id', '-timestamp']),
            models.Index(fields=['logistics_status']),
        ]
    
    def __str__(self):
        return f"{self.asset_id} - {self.timestamp} - {self.logistics_status}"


class ClassificationMetrics(models.Model):
    """
    Store classification model performance metrics over time for tracking and analysis
    """
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)
    accuracy = models.FloatField(default=0.0, help_text="Overall accuracy score (0-1)")
    precision = models.FloatField(default=0.0, help_text="Precision score (0-1)")
    recall = models.FloatField(default=0.0, help_text="Recall score (0-1)")
    f1_score = models.FloatField(default=0.0, help_text="F1 score (0-1)")
    total_samples = models.IntegerField(default=0, help_text="Total samples tested")
    data_source = models.CharField(
        max_length=50,
        choices=[
            ('csv_upload', 'CSV Upload'),
            ('pipeline_run', 'Pipeline Run'),
            ('manual_test', 'Manual Test'),
        ],
        default='manual_test'
    )
    confusion_matrix_json = models.JSONField(
        default=dict,
        help_text="Confusion matrix stored as JSON"
    )
    class_labels = models.JSONField(
        default=list,
        help_text="Classification labels"
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional notes about the run"
    )
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Classification Metrics'
        verbose_name_plural = 'Classification Metrics'
        indexes = [
            models.Index(fields=['-timestamp']),
            models.Index(fields=['data_source']),
        ]
    
    def __str__(self):
        return f"Classification Run - {self.timestamp} - Accuracy: {self.accuracy:.2%}"
