from rest_framework import viewsets
from rest_framework.filters import SearchFilter, OrderingFilter
from .models import ProcessLog
from .serializers import ProcessLogSerializer

class ProcessLogViewSet(viewsets.ModelViewSet):
    queryset = ProcessLog.objects.all()
    serializer_class = ProcessLogSerializer
    filter_backends = [SearchFilter, OrderingFilter]
    search_fields = ['message', 'process__process']
    ordering_fields = ['timestamp', 'process']
