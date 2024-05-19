from django.urls import path, include
from rest_framework.routers import DefaultRouter
from databrickschatbotapi.DataSciencePipelineProcess.views import DataSciencePipelineProcessViewSet
from databrickschatbotapi.DataSciencePipelineResult.views import DataSciencePipelineResultViewSet
from databrickschatbotapi.DatasetandInput.views import DatasetandInputViewSet
from databrickschatbotapi.ProcessLog.views import ProcessLogViewSet
from databrickschatbotapi.ProblemType.views import ProblemTypeViewSet
from databrickschatbotapi.views import RunPipelineView,ProcessLogsView,GeneratedDocumentView,ChatbotServiceView
router = DefaultRouter()
router.register(r'DataSciencePipelineProcess', DataSciencePipelineProcessViewSet)
router.register(r'DatasetandInput', DatasetandInputViewSet)
router.register(r'ProcessLog', ProcessLogViewSet)
router.register(r'DataSciencePipelineResult', DataSciencePipelineResultViewSet)
router.register(r'ProblemType', ProblemTypeViewSet)
#router.register(r'api',RunPipelineView)

urlpatterns = [
    path('', include(router.urls)),
    path('run-pipeline/', RunPipelineView.as_view(), name='run-pipeline'),
    path('get-processlogs/', ProcessLogsView.as_view(), name='get-processlogs'),
    path('get-analysisreport/', GeneratedDocumentView.as_view(), name='get-analysisreport'),
    path('chatwith-agent/', ChatbotServiceView.as_view(), name='chatwith-agent'),

]
