from celery import Celery

app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

app.conf.beat_schedule = {
    'retrain-models-every-day': {
        'task': 'tasks.retrain_models',
        'schedule': 86400.0,  # 24 hours
    },
}

@app.task
def retrain_models():
    from retrain_models import retrain_symptom_checker, retrain_predictive_analysis
    retrain_symptom_checker()
    retrain_predictive_analysis()