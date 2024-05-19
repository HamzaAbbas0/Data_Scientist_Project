from rest_framework import serializers
from .models import DataSciencePipelineResult

class DataSciencePipelineResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSciencePipelineResult
        fields = '__all__'
