from django.urls import path
from .views import RunPipelineView

urlpatterns = [
    # ... your other url patterns
    path('run-pipeline/', RunPipelineView.as_view(), name='run-pipeline'),
]
