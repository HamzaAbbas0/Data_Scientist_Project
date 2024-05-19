from rest_framework import serializers
from .models import DatasetandInput

class DatasetandInputSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetandInput
        fields = '__all__'