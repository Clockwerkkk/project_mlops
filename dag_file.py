import mlflow
import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from typing import Dict, Any
from datetime import datetime, timedelta


BUCKET = Variable.get("S3_BUCKET")

DEFAULT_ARGS = {
    'owner': 'igorpoletaev',  # Ваше ФИ
    'depends_on_past': False,  # Не зависит от прошлых запусков
    'email_on_failure': False,  # Отключаем уведомления о сбоях
    'email_on_retry': False,  # Отключаем уведомления при ретраях
    'retries': 3,  # Три попытки при сбое
    'retry_delay': timedelta(minutes=1),  # Задержка между ретраями — 1 минута
}

model_names = ["random_forest", "linear_regression", "decision_tree"]
models = {
    'linear_regression': LinearRegression(),
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
}

def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def init(**kwargs) -> Dict[str, Any]:
    configure_mlflow()
    experiment_name = "igor_poletaev"
    
    # Установка или создание эксперимента
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    
    # Начало родительского run
    with mlflow.start_run(run_name="Cl0ckwerkkk") as parent_run:
        run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id
        
        # Логируем параметры эксперимента
        mlflow.log_param("experiment_id", experiment_id)
        mlflow.log_param("run_id", run_id)
        
        # Передача метрик между шагами
        return {
            'experiment_id': experiment_id,
            'run_id': run_id
        }

def get_data(**kwargs) -> Dict[str, Any]:
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

def prepare_data(**kwargs) -> Dict[str, Any]:
    X_train = kwargs['ti'].xcom_pull(task_ids='get_data')['X_train']
    X_test = kwargs['ti'].xcom_pull(task_ids='get_data')['X_test']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled
    }

def train_model(**kwargs) -> Dict[str, Any]:
    model_name = kwargs['model_name']
    experiment_id = kwargs['ti'].xcom_pull(task_ids='init')['experiment_id']
    run_id = kwargs['ti'].xcom_pull(task_ids='init')['run_id']

    X_train = kwargs['ti'].xcom_pull(task_ids='prepare_data')['X_train_scaled']
    X_test = kwargs['ti'].xcom_pull(task_ids='prepare_data')['X_test_scaled']
    y_train = kwargs['ti'].xcom_pull(task_ids='get_data')['y_train']
    y_test = kwargs['ti'].xcom_pull(task_ids='get_data')['y_test']
    
    model = models[model_name]
    
    # Логирование дочерних run
    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id, parent_run_id=run_id, nested=True) as child_run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Логируем метрики
        mse = ((y_test - y_pred) ** 2).mean()
        mlflow.log_metric("mse", mse)
        
        # Логируем модель
        mlflow.sklearn.log_model(model, f"models/{model_name}")
        
        return {
            'model_name': model_name,
            'mse': mse
        }

def save_results(**kwargs):
    results = kwargs['ti'].xcom_pull(task_ids=['train_model_linear_regression', 'train_model_decision_tree', 'train_model_random_forest'])


with DAG(
    dag_id="igor_poletaev",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(1),
    catchup=False,
    tags=["mlops"]
) as dag:

    task_init = PythonOperator(
        task_id="init",
        python_callable=init,
        provide_context=True,
    )

    task_get_data = PythonOperator(
        task_id="get_data",
        python_callable=get_data,
        provide_context=True,
    )

    task_prepare_data = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data,
        provide_context=True,
    )
    
    training_model_tasks = []
    for model_name in models.keys():
        train_task = PythonOperator(
            task_id=f'train_model_{model_name}',
            python_callable=train_model,
            op_kwargs={'model_name': model_name},
            provide_context=True,
        )
        training_model_tasks.append(train_task)

    task_save_results = PythonOperator(
        task_id="save_results",
        python_callable=save_results,
        provide_context=True,
    )





#task_init = #

#task_get_data = #

#task_prepare_data = #

#training_model_tasks = [#]

#task_save_results = #

task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results
