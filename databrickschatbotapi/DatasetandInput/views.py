from rest_framework import viewsets, filters
from .models import DatasetandInput
from .serializers import DatasetandInputSerializer
from rest_framework.parsers import MultiPartParser, FormParser
class DatasetandInputViewSet(viewsets.ModelViewSet):
    queryset = DatasetandInput.objects.all()
    serializer_class = DatasetandInputSerializer
    filter_backends = (filters.SearchFilter, filters.OrderingFilter)
    search_fields = ['target_variable', 'problem_type__type']
    ordering_fields = ['id', 'target_variable']
    parser_classes = (MultiPartParser, FormParser)  # Add this to handle file uploads

    def perform_create(self, serializer):
        serializer.save(file=self.request.data.get('file'))
        