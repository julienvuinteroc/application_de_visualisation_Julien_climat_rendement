from prefect import task
from pipelines.load_data import load_all
from pipelines.load_projections import load_projections

@task
def load_data_task():
    load_all()

@task
def load_projection_task():
    load_projections()
