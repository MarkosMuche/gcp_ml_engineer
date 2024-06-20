

from google_cloud_pipeline_components import aiplatform as gcc_aip
from kfp.v2 import dsl
import os

import datetime
from dotenv import load_dotenv
load_dotenv()

PROJECT_ID = os.environ.get('PROJECT_ID')
REGION = "us-central1"
GCS_BUCKET = "gs://vetexai-bucket-test"
ARTIFACT_REGISTRY="bert-sentiment-classifier"

TIMESTAMP=datetime.datetime.now().strftime('%Y%m%d%H%M%S')
DISPLAY_NAME = "bert-sentiment-{}".format(TIMESTAMP)
GCS_BASE_OUTPUT_DIR= f"{GCS_BUCKET}/bert_sentiment_classifier-{TIMESTAMP}"
SERVING_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-11:latest"
IMAGE_NAME="bert-sentiment-classifier"
IMAGE_TAG="latest"
IMAGE_URI=f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{ARTIFACT_REGISTRY}/{IMAGE_NAME}:{IMAGE_TAG}"
USER = "MarkosMuche" 
PIPELINE_ROOT = "{}/pipeline_root/{}".format(GCS_BUCKET, USER)


@dsl.pipeline(name="bert-sentiment-classification", pipeline_root=PIPELINE_ROOT)
def pipeline(
    project: str = PROJECT_ID,
    location: str = REGION,
    staging_bucket: str = GCS_BUCKET,
    display_name: str = DISPLAY_NAME,    
    container_uri: str = IMAGE_URI,
    model_serving_container_image_uri: str = SERVING_IMAGE_URI,    
    base_output_dir: str = GCS_BASE_OUTPUT_DIR,
):
    
    model_train_evaluate_op = gcc_aip.CustomContainerTrainingJobRunOp(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        replica_count=1,
        machine_type="e2-standard-4",
        display_name=display_name,
        container_uri=container_uri,
        base_output_dir=base_output_dir,
        
        model_serving_container_image_uri=model_serving_container_image_uri,
    )    
    
    # Create a Vertex Endpoint resource in parallel with model training.
    endpoint_create_op = gcc_aip.EndpointCreateOp(
        # Vertex AI Python SDK authentication parameters.
        project=project,
        location=location,
        display_name=display_name
    
    )   
    
    # Deploy your model to the created Endpoint resource for online predictions.
    model_deploy_op = gcc_aip.ModelDeployOp(
        # Link to model training component through output model artifact.
        model=model_train_evaluate_op.outputs["model"],
        # Link to the created Endpoint.
        endpoint=endpoint_create_op.outputs["endpoint"],
        # Define prediction request routing. {"0": 100} indicates 100% of traffic 
        # to the ID of the current model being deployed.
        traffic_split={"0": 100},
        # WorkerPool arguments.        
        dedicated_resources_machine_type="e2-standard-4",
        dedicated_resources_min_replica_count=1,
        dedicated_resources_max_replica_count=2
    )
