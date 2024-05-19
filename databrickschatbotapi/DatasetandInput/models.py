from django.db import models

from databrickschatbotapi.ProblemType.models import ProblemType

class DatasetandInput(models.Model):
    id = models.AutoField(primary_key=True)
    file = models.FileField(upload_to='datasets/')
    target_variable = models.CharField(max_length=100)
    #problemtype = models.CharField(max_length=100)
    datetime_column =  models.CharField(max_length=100, blank=True, null=True)
    problem_type = models.CharField(max_length=100, blank=True, null=True)
    # Foreign key to ProblemType
#     problem_type = models.ForeignKey(
#         ProblemType,
#         on_delete=models.SET_NULL,  # Set to NULL if the ProblemType is deleted
#         null=True,                 # This field is optional
#         related_name='datasets'
#     )

    def __str__(self):
        return f"{self.file.name} ({self.problem_type if self.problem_type else 'No Type'}) - Input ID: {self.id}"
