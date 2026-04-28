from prefect import flow

from pipelines.tasks.load_tasks import load_data_task, load_projection_task
from pipelines.tasks.clean_tasks import clean_task
from pipelines.tasks.aggregate_tasks import aggregate_task
from pipelines.tasks.merge_tasks import merge_task, build_task

@flow(name="pipeline_mvttdb")
def main_flow():

    # ordre logique
    load_data_task()
    load_projection_task()

    clean_task()
    aggregate_task()

    merge_task()
    build_task()

    print("🎉 PIPELINE ORCHESTRÉ TERMINÉ")

if __name__ == "__main__":
    main_flow()
