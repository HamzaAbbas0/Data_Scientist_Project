# Generated by Django 4.2.3 on 2024-05-06 08:17

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DataSciencePipelineProcess',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('process', models.TextField()),
                ('datetime', models.DateTimeField(auto_now=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('running', 'Running'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=10)),
            ],
        ),
        migrations.CreateModel(
            name='DatasetandInput',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('file', models.FileField(upload_to='datasets/')),
                ('target_variable', models.CharField(max_length=100)),
                ('datetime_column', models.CharField(blank=True, max_length=100, null=True)),
                ('problem_type', models.CharField(blank=True, max_length=100, null=True)),
            ],
        ),
        migrations.CreateModel(
            name='ProblemType',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('type', models.CharField(choices=[('timeseries', 'Time Series'), ('prediction', 'Prediction'), ('classification', 'Classification')], max_length=15, unique=True)),
                ('description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='ProcessLog',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('timestamp', models.DateTimeField(auto_now_add=True)),
                ('message', models.TextField()),
                ('process', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='logs', to='databrickschatbotapi.datasciencepipelineprocess')),
            ],
        ),
        migrations.CreateModel(
            name='DataSciencePipelineResult',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('result_data', models.JSONField()),
                ('generatedreportfile_path', models.CharField(max_length=1024)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('process', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='results', to='databrickschatbotapi.datasciencepipelineprocess')),
            ],
        ),
        migrations.AddField(
            model_name='datasciencepipelineprocess',
            name='inputinfo',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='process', to='databrickschatbotapi.datasetandinput'),
        ),
    ]