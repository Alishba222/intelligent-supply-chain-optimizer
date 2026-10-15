from django.core.management.base import BaseCommand
from django.utils import timezone
from analytics.models import RealTimeAssetData
from datetime import datetime, timedelta
import random


class Command(BaseCommand):
    help = 'Load sample real-time asset data into the database'

    def add_arguments(self, parser):
        parser.add_argument(
            '--count',
            type=int,
            default=50,
            help='Number of sample records to create (default: 50)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear all existing records before loading new ones'
        )

    def handle(self, *args, **options):
        count = options['count']
        
        if options['clear']:
            deleted_count, _ = RealTimeAssetData.objects.all().delete()
            self.stdout.write(
                self.style.WARNING(f'Deleted {deleted_count} existing records')
            )

        # Sample data
        assets = [f'Truck {i}' for i in range(1, 11)]  # Truck 1-10
        regions = [
            'Southeast Asia',
            'East Asia',
            'South Asia',
            'Europe',
            'North America'
        ]
        statuses = [
            'Advance shipping',
            'In Transit',
            'Delayed',
            'Delivered',
            'Pending'
        ]
        genres = [
            'Perishable',
            'Fragile',
            'Hazardous',
            'Electronics',
            'Textiles',
            'N/A'
        ]

        created_records = []
        base_time = timezone.now() - timedelta(days=7)

        for i in range(count):
            timestamp = base_time + timedelta(hours=random.randint(0, 168))
            
            record = RealTimeAssetData(
                timestamp=timestamp,
                asset_id=random.choice(assets),
                operational_region=random.choice(regions),
                logistics_status=random.choice(statuses),
                iot_temperature=round(random.uniform(15.0, 35.0), 1),
                traffic_volume_scale=round(random.uniform(10, 100), 1),
                waiting_latency_hours=round(random.uniform(0.5, 48.0), 2),
                nlp_classified_genre=random.choice(genres)
            )
            created_records.append(record)

        # Bulk create
        RealTimeAssetData.objects.bulk_create(created_records)

        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully created {len(created_records)} sample real-time asset data records'
            )
        )

        # Print summary statistics
        total_count = RealTimeAssetData.objects.count()
        assets_in_db = RealTimeAssetData.objects.values_list('asset_id', flat=True).distinct()
        regions_in_db = RealTimeAssetData.objects.values_list('operational_region', flat=True).distinct()

        self.stdout.write(
            self.style.SUCCESS(
                f'\nDatabase Summary:\n'
                f'  Total Records: {total_count}\n'
                f'  Unique Assets: {len(set(assets_in_db))}\n'
                f'  Unique Regions: {len(set(regions_in_db))}'
            )
        )
