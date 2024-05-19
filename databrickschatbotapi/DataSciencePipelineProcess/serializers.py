from rest_framework import serializers
from .models import (DataSciencePipelineProcess)

class DataSciencePipelineProcessSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSciencePipelineProcess
        fields = '__all__'