import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'databricks_chatbot_api.settings')

app = Celery('databricks_chatbot_api')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
#celery -A databricks_chatbot_api worker -l info

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
