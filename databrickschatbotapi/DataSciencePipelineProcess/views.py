from rest_framework import viewsets, filters
from .models import DataSciencePipelineProcess
from .serializers import DataSciencePipelineProcessSerializer
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone


class DataSciencePipelineProcessViewSet(viewsets.ModelViewSet):
    queryset = DataSciencePipelineProcess.objects.all()
    serializer_class = DataSciencePipelineProcessSerializer
    filter_backends = [filters.SearchFilter]
    search_fields = ['process', 'status','inputinfo']  # Allow filtering by process description or status
    ordering_fields = ['id', 'process', 'status','inputinfo']

    @action(detail=True, methods=['post'], url_path='update-status')
    def update_status(self, request, pk=None):
        """
        Custom action to update the status of a DataSciencePipelineProcess.
        """
        process = self.get_object()
        status = request.data.get('status', None)
        if status not in dict(DataSciencePipelineProcess.STATUS_CHOICES):
            return Response({'error': 'Invalid status'}, status=status.HTTP_400_BAD_REQUEST)

        process.status = status
        process.save()
        return Response({'status': 'Status updated to ' + status}, status=status.HTTP_200_OK)

    def get_queryset(self):
        """
        Optionally restricts the returned processes to a given user,
        by filtering against a `since` query parameter in the URL.
        """
        queryset = DataSciencePipelineProcess.objects.all()
        since = self.request.query_params.get('since', None)
        if since is not None:
            since_date = timezone.make_aware(datetime.datetime.strptime(since, '%Y-%m-%d'))
            queryset = queryset.filter(datetime__gte=since_date)
        return queryset
