from rest_framework import serializers
from .models import ProblemType

class ProblemTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProblemType
        fields = '__all__'