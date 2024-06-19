
from kfp import compiler, dsl

from google.cloud import aiplatform

from dotenv import load_dotenv
import os

load_dotenv

PROJECT_ID = os.environ['PROJECT_ID']

BUCKET_NAME = 'markosmuche-test'
PACKAGE_PATH = 'pipe.json'
REGION = 'us-central1'

@dsl.component
def say_hello(name: str) -> str:

    hello_text = f'hello {name}'
    print(hello_text)
    return hello_text


@dsl.pipeline
def hello_pipeline(recipient: str) ->str:

    hello_task = say_hello(name=recipient)

    return hello_task.output


# compiler.Compiler().compile(pipeline_conf=hello_pipeline,package_path='pipe.yaml')
compiler.Compiler().compile(pipeline_func=hello_pipeline, package_path=PACKAGE_PATH)
aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name='my-hello-pipeline',
    template_path=PACKAGE_PATH,
    pipeline_root=f'gs://{BUCKET_NAME}',
    parameter_values=dict(recipient = 'Markos Muche')
)

job.run()