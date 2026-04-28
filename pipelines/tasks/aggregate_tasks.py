from prefect import task
from utils.db import get_connection
from pipelines.aggregate_climat import aggregate

@task
def aggregate_task():
    con = get_connection()
    aggregate(con)
