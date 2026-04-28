from prefect import task
from utils.db import get_connection
from pipelines.clean_data import clean

@task
def clean_task():
    con = get_connection()
    clean(con)
