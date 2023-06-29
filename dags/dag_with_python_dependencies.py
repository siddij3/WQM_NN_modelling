from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    'owner': 'coder2j',
    'retry': 5,
    'retry_delay': timedelta(minutes=5)
}


def get_dependencies():
    import sklearn
    import tensorflow
    import keras
    import pandas
    print(f"sklearn with version: {sklearn.__version__} ")
    print(f"tensorflow with version: {tensorflow.__version__} ")
    print(f"keras with version: {keras.__version__} ")
    print(f"pandas with version: {pandas.__version__} ")


def get_branch_dependencies():
    import cloudinary
    import sqlalchemy as sa
    print(f"sqlalchemy with version: {sa.__version__} ")



with DAG(
    default_args=default_args,
    dag_id="dag_with_python_dependencies_v03",
    start_date=datetime(2021, 10, 12),
    schedule_interval='@daily'
) as dag:
    task1 = PythonOperator(
        task_id='get_main_dependencies',
        python_callable=get_dependencies
    )
    
    task2 = PythonOperator(
        task_id='get_branch_dependencies',
        python_callable=get_branch_dependencies
    )

    task1 >> task2