from prefect import task
from utils.db import get_connection
from pipelines.merge_datasets import merge
from pipelines.build_features import build

@task
def merge_task():
    con = get_connection()
    merge(con)

@task
def build_task():
    con = get_connection()
    build(con)
