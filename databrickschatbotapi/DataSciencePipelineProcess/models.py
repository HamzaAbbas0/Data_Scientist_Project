from django.db import models
from databrickschatbotapi.DatasetandInput.models import DatasetandInput

class DataSciencePipelineProcess(models.Model):
    # A unique identifier for the process. Django automatically adds an `id` field if you don't specify one,
    # but it's good practice to define it explicitly when discussing model structure.
    id = models.AutoField(primary_key=True)

    # A text field to describe what the data science process is about.
    process = models.TextField()

    # A datetime field that stores the last updated or checked timestamp.
    # `auto_now=True` makes the field automatically update to the current timestamp when the object is saved.
    datetime = models.DateTimeField(auto_now=True)
    inputinfo = models.ForeignKey(
        DatasetandInput,
        on_delete=models.CASCADE,
        related_name='process'
    )
    # A choice field to indicate the status of the process.
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed')
    ]
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')

    def __str__(self):
        return f"{self.process} ({self.status})({self.inputinfo.id})"