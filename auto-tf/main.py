import boto3
import json
import os
import joblib
import pickle
import tarfile
import sagemaker
from sagemaker.estimator import Estimator
import time
from time import gmtime, strftime
import subprocess


#Setup
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
print(region)
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::474422712127:role/sagemaker-role-BYOC"

#Build tar file with model data + inference code
#ToDo: Edit this
#bashCommand = "tar -cvpzf model.tar.gz 0000001 inference.py"
#process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
#output, error = process.communicate()


# retrieve tf image
image_uri = sagemaker.image_uris.retrieve(
    framework="tensorflow",
    region="us-east-1",
    version="2.3.0",
    py_version="py3",
    instance_type="ml.m5.xlarge",
    image_scope="inference"
)

#Bucket for model artifacts
default_bucket = sagemaker_session.default_bucket()
print(default_bucket)

#Upload tar.gz to bucket
model_artifacts = f"s3://{default_bucket}/model.tar.gz"
response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')


#Step 1: Model Creation
model_name = "tf-test" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
print("Model name: " + model_name)
create_model_response = client.create_model(
    ModelName=model_name,
    Containers=[
        {
            "Image": image_uri,
            "Mode": "SingleModel",
            "ModelDataUrl": model_artifacts,
            "Environment": {'SAGEMAKER_SUBMIT_DIRECTORY': model_artifacts,
                            'SAGEMAKER_PROGRAM': 'inference.py'} 
        }
    ],
    ExecutionRoleArn=role,
)
print("Model Arn: " + create_model_response["ModelArn"])


#Step 2: EPC Creation
epc_name = "tf-epc" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
endpoint_config_response = client.create_endpoint_config(
    EndpointConfigName=epc_name,
    ProductionVariants=[
        {
            "VariantName": "tfvariant",
            "ModelName": model_name,
            "InstanceType": "ml.c5.large",
            "InitialInstanceCount": 1
        },
    ],
)
print("Endpoint Configuration Arn: " + endpoint_config_response["EndpointConfigArn"])


#Step 3: EP Creation
endpoint_name = "tf-ep" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
create_endpoint_response = client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=epc_name,
)
print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])


#Monitor creation
describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
while describe_endpoint_response["EndpointStatus"] == "Creating":
    describe_endpoint_response = client.describe_endpoint(EndpointName=endpoint_name)
    print(describe_endpoint_response["EndpointStatus"])
    time.sleep(15)
print(describe_endpoint_response)