from rest_framework import views, status
from rest_framework.response import Response
from databrickschatbotapi.tasks import run_pipeline_async
from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from databrickschatbotapi.DataSciencePipelineResult.models import DataSciencePipelineResult
from databrickschatbotapi.DataSciencePipelineResult.models import DataSciencePipelineProcess
from databrickschatbotapi.ProcessLog.models import ProcessLog
from databrickschatbotapi.DatasetandInput.serializers import DatasetandInputSerializer
from databrickschatbotapi.DataSciencePipelineProcess.serializers import DataSciencePipelineProcessSerializer
from databrickschatbotapi.DatasetandInput.models import DatasetandInput
import os
from django.http import FileResponse
from rest_framework.parsers import MultiPartParser, FormParser
from databrickschatbotapi.DatascientistPipeline.datasciencepipeline import process_chatbot_query
class RunPipelineView(APIView):
    parser_classes = (MultiPartParser, FormParser)  # Enables handling of multipart form data

    def post(self, request, *args, **kwargs):
        dataset_serializer = DatasetandInputSerializer(data=request.data)
        if dataset_serializer.is_valid():
            dataset_instance = dataset_serializer.save()
            
            # Extracting details from dataset_instance
            folder_path = dataset_instance.file.path.rsplit('/', 1)[0]
            file_name = dataset_instance.file.name
            problem_type = dataset_instance.problem_type
            target_datacolumn = dataset_instance.target_variable if dataset_instance.target_variable else None
            date_column = dataset_instance.datetime_column if dataset_instance.datetime_column else None

            # Create a DataSciencePipelineProcess instance
            process_instance = DataSciencePipelineProcess.objects.create(
                inputinfo=dataset_instance,
                process= file_name+' is'+'provided to process as problem type '+ problem_type,
                status='pending'
            )
            process_serializer = DataSciencePipelineProcessSerializer(process_instance)
            if problem_type == "DocumentAnalysis":
                return Response({"message": "Document Process has been Registered", "process_id": process_instance.id}, status=status.HTTP_202_ACCEPTED)
            elif problem_type == "ImageAnalysis":
                return Response({"message": "Image Process has been Registered", "process_id": process_instance.id}, status=status.HTTP_202_ACCEPTED)                
            else:

                # Asynchronous pipeline execution with specific fields and process ID
                inputcolumns=run_pipeline_async( 
                    file_name, 
                    problem_type, 
                    target_datacolumn, 
                    date_column, 
                    process_instance.id  # Include process ID
                )
                if problem_type == "numerical" or problem_type == "categorical":
                    inputcolumns = inputcolumns
                else:
                    inputcolumns = None

                    
                return Response({"message": "Pipeline is running in the background", "process_id": process_instance.id,"inputcolumns":inputcolumns}, status=status.HTTP_202_ACCEPTED)
        else:
            return Response(dataset_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ProcessLogsView(APIView):
    def get(self, request):
        # Retrieve the process_id from the query parameters
        process_id = request.query_params.get('process_id')
        if not process_id:
            return Response({'error': 'process_id parameter is required'}, status=status.HTTP_400_BAD_REQUEST)

        # Filter the logs based on the retrieved process_id
        logs = ProcessLog.objects.filter(process__id=process_id)
        if not logs.exists():
            return Response({'error': 'No logs found for the given process_id'}, status=status.HTTP_404_NOT_FOUND)

        logs_data = [{"message": log.message, "timestamp": log.timestamp.isoformat()} for log in logs]
        return Response(logs_data, status=status.HTTP_200_OK)


class GeneratedDocumentView(APIView):
    
    def get(self, request, process_id):
        try:
            # Retrieve the result by the related_name 'results' in ForeignKey
            document = DataSciencePipelineResult.objects.get(process__id=process_id)
        except DataSciencePipelineResult.DoesNotExist:
            # If no document is found, return a 404 Not Found response
            return Response({"error": "Document not found"}, status=status.HTTP_404_NOT_FOUND)

        # Check if the file exists on the server
        if os.path.exists(document.generatedreportfile_path):
            # If the file exists, return it as a file response
            return FileResponse(open(document.generatedreportfile_path, 'rb'), content_type='application/pdf')
        else:
            # If the file does not exist, return a 404 Not Found response
            return Response({"error": "File does not exist"}, status=status.HTTP_404_NOT_FOUND)


class ChatbotServiceView(APIView):
                            
    def post(self, request):
        
        query = request.data.get("query")
        print(query)
        process_id = request.data.get("process_id")
        if not query:
            return Response({"error": "No query provided"}, status=status.HTTP_400_BAD_REQUEST)
        response = process_chatbot_query(query,process_id) # Assume this is your function to process queries
        return Response({"response": response}, status=status.HTTP_200_OK)
