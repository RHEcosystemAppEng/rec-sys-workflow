from kfp import dsl, compiler, Client
from typing import List, Dict
import os
from kfp import kubernetes
from kfp.dsl import Input, Output, Dataset, Model, Artifact

IMAGE_TAG = '0.0.43'
BASE_IMAGE = os.getenv("BASE_REC_SYS_IMAGE", f"quay.io/ecosystem-appeng/rec-sys-app:{IMAGE_TAG}")

@dsl.component(base_image=BASE_IMAGE)
def generate_candidates(item_input_model: Input[Model], user_input_model: Input[Model], item_df_input: Input[Dataset], user_df_input: Input[Dataset], models_definition_input: Input[Artifact]):
    from feast import FeatureStore
    from feast.data_source import PushMode
    from models.data_util import data_preproccess
    from models.entity_tower import EntityTower
    from service.clip_encoder import ClipEncoder
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import torch
    import subprocess
    import json

    with open(models_definition_input.path, 'r') as f:
        models_definition :dict = json.load(f)

    result = subprocess.run(
        ["/bin/bash", "-c", "ls && ./entry_point.sh"],
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Return output as strings (not bytes)
        # check=True           # Raise an error if the command fails
    )

    # Print the stdout
    print("Standard Output:")
    print(result.stdout)

    # Print the stderr (if any)
    print("Standard Error:")
    print(result.stderr)
    with open('feature_repo/feature_store.yaml', 'r') as file:
        print(file.read())

    store = FeatureStore(repo_path="feature_repo/")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    item_encoder = EntityTower(models_definition['items_num_numerical'], models_definition['items_num_categorical'])
    user_encoder = EntityTower(models_definition['users_num_numerical'], models_definition['users_num_categorical'])
    item_encoder.load_state_dict(torch.load(item_input_model.path))
    user_encoder.load_state_dict(torch.load(user_input_model.path))
    item_encoder.to(device)
    user_encoder.to(device)
    item_encoder.eval()
    user_encoder.eval()
    # load item and user dataframes
    item_df = pd.read_parquet(item_df_input.path)
    user_df = pd.read_parquet(user_df_input.path)

    # Create a new table to be push to the online store
    item_embed_df = item_df[['item_id']].copy()
    user_embed_df = user_df[['user_id']].copy()

    # Encode the items and users
    proccessed_items = data_preproccess(item_df)
    proccessed_users = data_preproccess(user_df)
    # Move tensors to device
    proccessed_items = {key: value.to(device) if type(value) == torch.Tensor else value for key, value in proccessed_items.items()}
    proccessed_users = {key: value.to(device) if type(value) == torch.Tensor else value for key, value in proccessed_users.items()}
    item_embed_df['embedding'] = item_encoder(**proccessed_items).detach().numpy().tolist()
    user_embed_df['embedding'] = user_encoder(**proccessed_users).detach().numpy().tolist()

    # Add the currnet timestamp
    item_embed_df['event_timestamp'] = datetime.now()
    user_embed_df['event_timestamp'] = datetime.now()

    # Push the new embedding to the offline and online store
    store.push('item_embed_push_source', item_embed_df, to=PushMode.ONLINE, allow_registry_cache=False)
    store.push('user_embed_push_source', user_embed_df, to=PushMode.ONLINE, allow_registry_cache=False)

    # Store the embedding of text features for search by text
    item_text_features_embed = item_df[['item_id']].copy()
    # item_text_features_embed['product_name'] = proccessed_items['text_features'].detach()[:, 0, :].numpy().tolist()
    item_text_features_embed['about_product_embedding'] = proccessed_items['text_features'].detach()[:, 1, :].numpy().tolist()
    item_text_features_embed['event_timestamp'] = datetime.now()

    store.push('item_textual_features_embed', item_text_features_embed, to=PushMode.ONLINE, allow_registry_cache=False)

    # Store the embedding of clip features for search by image
    clip_encoder = ClipEncoder()
    item_clip_features_embed = clip_encoder.clip_embeddings(item_df)
    store.push('item_clip_features_embed', item_clip_features_embed, to=PushMode.ONLINE, allow_registry_cache=False)

    # Materilize the online store
    store.materialize_incremental(datetime.now(), feature_views=['item_embedding', 'user_items', 'item_features', 'item_textual_features_embed'])

    # Calculate user recommendations for each user
    item_embedding_view = 'item_embedding'
    k = 64
    item_recommendation = []
    for user_embed in user_embed_df['embedding']:
        item_recommendation.append(
            store.retrieve_online_documents(
                query=user_embed,
                top_k=k,
                features=[f'{item_embedding_view}:item_id']
            ).to_df()['item_id'].to_list()
        )

    # Pushing the calculated items to the online store
    user_items_df = user_embed_df[['user_id']].copy()
    user_items_df['event_timestamp'] = datetime.now()
    user_items_df['top_k_item_ids'] = item_recommendation

    store.push('user_items_push_source', user_items_df, to=PushMode.ONLINE, allow_registry_cache=False)


@dsl.component(base_image=BASE_IMAGE, packages_to_install=['minio'])
def train_model(item_df_input: Input[Dataset], user_df_input: Input[Dataset], interaction_df_input: Input[Dataset], item_output_model: Output[Model], user_output_model: Output[Model], models_definition_output: Output[Artifact]):
    from models.train_two_tower import create_and_train_two_tower
    import pandas as pd
    import torch
    import json
    import os
    from minio import Minio

    item_df = pd.read_parquet(item_df_input.path)
    user_df = pd.read_parquet(user_df_input.path)
    interaction_df = pd.read_parquet(interaction_df_input.path)

    item_encoder, user_encoder, models_definition= create_and_train_two_tower(item_df, user_df, interaction_df, return_model_definition=True)

    torch.save(item_encoder.state_dict(), item_output_model.path)
    torch.save(user_encoder.state_dict(), user_output_model.path)
    item_output_model.metadata['framework'] = 'pytorch'
    user_output_model.metadata['framework'] = 'pytorch'
    with open(models_definition_output.path, 'w') as f:
        json.dump(models_definition, f)
    
    minio_client = Minio(
        endpoint=os.getenv('MINIO_HOST', "endpoint") + \
            ':' + os.getenv('MINIO_PORT', '9000'),
        access_key=os.getenv('MINIO_ACCESS_KEY', "access-key"),
        secret_key=os.getenv('MINIO_SECRET_KEY', "secret-key"),
        secure=False  # Set to True if using HTTPS
    )
    
    bucket_name = "user-encoder"
    object_name = "user-encoder.pth" 
    configuration = 'user-encoder-config.json'
    
    # Ensure the bucket exists, create it if it doesn't
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)

    minio_client.fput_object(
        bucket_name=bucket_name,
        object_name=object_name,
        file_path=user_output_model.path
    )
    # Save model configurations
    minio_client.fput_object(
        bucket_name=bucket_name,
        object_name=configuration,
        file_path=models_definition_output.path
    )


@dsl.component(base_image=BASE_IMAGE, packages_to_install=['psycopg2'])
def load_data_from_feast(item_df_output: Output[Dataset], user_df_output: Output[Dataset], interaction_df_output: Output[Dataset]):
    from feast import FeatureStore
    import pandas as pd
    import os
    from service.dataset_provider import LocalDatasetProvider
    from sqlalchemy import create_engine, text
    import subprocess

    result = subprocess.run(
        ["/bin/bash", "-c", "ls && ./entry_point.sh"],
        capture_output=True,  # Capture stdout and stderr
        text=True,           # Return output as strings (not bytes)
    )

    # Print the stdout
    print("Standard Output:")
    print(result.stdout)

    # Print the stderr (if any)
    print("Standard Error:")
    print(result.stderr)

    with open('feature_repo/feature_store.yaml', 'r') as file:
        print(file.read())
    store = FeatureStore(repo_path="feature_repo/")
    store.refresh_registry()
    print('registry refreshed')
    dataset_provider = LocalDatasetProvider(store)

    # retrieve datasets for training
    item_df = dataset_provider.item_df()
    user_df = dataset_provider.user_df()
    interaction_df = dataset_provider.interaction_df()

    uri = os.getenv('uri', None)
    engine = create_engine(uri)

    def table_exists(engine, table_name):
        query = text("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = :table_name")
        with engine.connect() as connection:
            result = connection.execute(query, {"table_name": table_name}).scalar()
            return result > 0

    if table_exists(engine, 'new_users'):
        query_new_users = 'SELECT * FROM new_users'
        stream_users_df = pd.read_sql(query_new_users, engine).rename(columns={'timestamp':'signup_date'})

        user_df = pd.concat([user_df, stream_users_df], axis=0)

    if table_exists(engine, 'stream_interaction'):
        query_positive = 'SELECT * FROM stream_interaction'
        stream_positive_inter_df = pd.read_sql(query_positive, engine).rename(columns={'timestamp':'event_timestamp'})

        interaction_df = pd.concat([interaction_df, stream_positive_inter_df], axis=0)

    # Pass artifacts
    item_df.to_parquet(item_df_output.path)
    user_df.to_parquet(user_df_output.path)
    interaction_df.to_parquet(interaction_df_output.path)

    item_df_output.metadata['format'] = 'parquet'
    user_df_output.metadata['format'] = 'parquet'
    interaction_df_output.metadata['format'] = 'parquet'


def mount_secret_feast_repository(task):
    kubernetes.use_secret_as_env(
        task=task,
        secret_name=os.getenv('DB_SECRET_NAME', 'cluster-sample-app'),
        secret_key_to_env={
            'uri': 'uri',
            'password': 'DB_PASSWORD',
            'host': 'DB_HOST',
            'dbname': 'DB_NAME',
            'user': 'DB_USER',
            'port': 'DB_PORT',
        },
    )
    kubernetes.use_secret_as_volume(
        task=task,
        secret_name=os.getenv("FEAST_SECRET_NAME", 'feast-feast-rec-sys-registry-tls'),
        mount_path='/app/feature_repo/secrets',
    )
    task.set_env_variable(name="FEAST_PROJECT_NAME", value=os.getenv("FEAST_PROJECT_NAME", "feast_rec_sys"))
    task.set_env_variable(name="FEAST_REGISTRY_URL", value=os.getenv("FEAST_REGISTRY_URL", "feast-feast-rec-sys-registry.rec-sys.svc.cluster.local"))

@dsl.pipeline(name=os.path.basename(__file__).replace(".py", ""))
def batch_recommendation():

    load_data_task = load_data_from_feast()
    mount_secret_feast_repository(load_data_task)
    # Component configurations
    load_data_task.set_caching_options(False)

    train_model_task = train_model(
        item_df_input=load_data_task.outputs['item_df_output'],
        user_df_input=load_data_task.outputs['user_df_output'],
        interaction_df_input=load_data_task.outputs['interaction_df_output'],
    ).after(load_data_task)
    train_model_task.set_caching_options(False)
    kubernetes.use_secret_as_env(
        task=train_model_task,
        secret_name=os.getenv('MINIO_SECRET_NAME', 'ds-pipeline-s3-dspa'),
        secret_key_to_env={
            'host': 'MINIO_HOST',
            'port': 'MINIO_PORT',
            'accesskey': 'MINIO_ACCESS_KEY',
            'secretkey': 'MINIO_SECRET_KEY',
            'secure': 'MINIO_SECURE',
        },
    )

    generate_candidates_task = generate_candidates(
        item_input_model=train_model_task.outputs['item_output_model'],
        user_input_model=train_model_task.outputs['user_output_model'],
        item_df_input=load_data_task.outputs['item_df_output'],
        user_df_input=load_data_task.outputs['user_df_output'],
        models_definition_input=train_model_task.outputs['models_definition_output'],
    ).after(train_model_task)
    kubernetes.use_secret_as_env(
        task=generate_candidates_task,
        secret_name=os.getenv('DB_SECRET_NAME', 'cluster-sample-app'),
        secret_key_to_env={
            'uri': 'uri',
            'password': 'DB_PASSWORD',
            'host': 'DB_HOST',
            'dbname': 'DB_NAME',
            'user': 'DB_USER',
            'port': 'DB_PORT',
        },
    )
    kubernetes.use_secret_as_volume(
        task=generate_candidates_task,
        secret_name=os.getenv("FEAST_SECRET_NAME", 'feast-feast-edb-rec-sys-registry-tls'),
        mount_path='/app/feature_repo/secrets',
    )
    generate_candidates_task.set_env_variable(name="FEAST_PROJECT_NAME", value=os.getenv("FEAST_PROJECT_NAME", "feast_edb_rec_sys"))
    generate_candidates_task.set_env_variable(name="FEAST_REGISTRY_URL", value=os.getenv("FEAST_REGISTRY_URL", "feast-feast-edb-rec-sys-registry.rec-sys.svc.cluster.local"))
    generate_candidates_task.set_caching_options(False)


if __name__ == "__main__":
    pipeline_yaml = __file__.replace(".py", ".yaml")

    compiler.Compiler().compile(
        pipeline_func=batch_recommendation,
        package_path=pipeline_yaml
    )

    client = Client(
      host=os.environ["DS_PIPELINE_URL"],
      verify_ssl=False
    )

    uploaded_pipeline = client.upload_pipeline(
      pipeline_package_path=pipeline_yaml,
      pipeline_name=os.environ["PIPELINE_NAME"]
    )

    run = client.create_run_from_pipeline_package(
      pipeline_file=pipeline_yaml,
      arguments={},
      run_name=os.environ["RUN_NAME"]
    )

    print(f"Pipeline submitted! Run ID: {run.run_id}")
