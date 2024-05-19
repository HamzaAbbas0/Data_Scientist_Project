from django.db import models

class ProblemType(models.Model):
    id = models.AutoField(primary_key=True)
    PROBLEM_CHOICES = [
        ('timeseries', 'Time Series'),
        ('prediction', 'Prediction'),
        ('classification', 'Classification')
    ]

    type = models.CharField(max_length=15, choices=PROBLEM_CHOICES, unique=True)
    description = models.TextField()

    def __str__(self):
        return self.type