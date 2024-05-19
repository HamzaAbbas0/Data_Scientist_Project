from rest_framework import viewsets, filters
from .serializers import DataSciencePipelineResultSerializer
from rest_framework.permissions import IsAuthenticated, DjangoModelPermissions
from .models import DataSciencePipelineResult

class DataSciencePipelineResultViewSet(viewsets.ModelViewSet):
    queryset = DataSciencePipelineResult.objects.all()
    serializer_class = DataSciencePipelineResultSerializer
    # permission_classes = [IsAuthenticated, DjangoModelPermissions]
