from rest_framework import viewsets, filters
from .models import ProblemType
from .serializers import ProblemTypeSerializer
from rest_framework.permissions import IsAuthenticated, DjangoModelPermissions
class ProblemTypeViewSet(viewsets.ModelViewSet):
    queryset = ProblemType.objects.all()
    serializer_class = ProblemTypeSerializer
    filter_backends = [filters.SearchFilter, filters.OrderingFilter]
    search_fields = ['type', 'description']
    ordering_fields = ['type', 'id']  # Allows clients to order by type or id
    # permission_classes = [IsAuthenticated, DjangoModelPermissions]
    # def get_queryset(self):
    #     """
    #     This can be customized to return different querysets based on user
    #     permissions or other logic like only showing active problem types.
    #     """
    #     if self.request.user.is_superuser:
    #         return ProblemType.objects.all()
    #     return ProblemType.objects.filter(is_active=True)


