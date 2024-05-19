from django.db import models
# Correct the import for JSONField
from databrickschatbotapi.DataSciencePipelineProcess.models import DataSciencePipelineProcess
import os

class DataSciencePipelineResult(models.Model):
    id = models.AutoField(primary_key=True)
    process = models.ForeignKey(
        DataSciencePipelineProcess,
        on_delete=models.CASCADE,
        related_name='results'
    )
    result_data = models.JSONField()  # Corrected JSONField import usage
    generatedreportfile_path = models.CharField(max_length=1024)  # Store full path to the file
    created_at = models.DateTimeField(auto_now_add=True)

    def delete(self, *args, **kwargs):
        if os.path.exists(self.generatedreportfile_path):
            os.remove(self.generatedreportfile_path)
        super().delete(*args, **kwargs)
        
    def __str__(self):
        return f"Results for Process ID {self.process.id} at {self.created_at}"
    
    @staticmethod
    def add_result(process_id, result_json, report_file_path):
        # Check if the given process exists
        try:
            process = DataSciencePipelineProcess.objects.get(id=process_id)

            # Check if the report file path exists
            if not os.path.exists(report_file_path):
                return False, "Report file path does not exist."

            # Create the result instance
            result = DataSciencePipelineResult(
                process=process,
                result_data=result_json,
                generatedreportfile_path=report_file_path
            )
            result.save()
            return True, "Result added successfully."
        except DataSciencePipelineProcess.DoesNotExist:
            return False, "Process with the given ID does not exist."



