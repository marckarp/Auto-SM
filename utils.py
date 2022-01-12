from platform import python_version
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

"""
Add code for

    1. Create Model
    2. Create Endpoint Config
    3. Create Endpoint
"""

#Setup
client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::474422712127:role/sagemaker-role-BYOC" #add functionality to add role


def retrieve_image(framework_type, instance_type, framework_version):
    image_uri = sagemaker.image_uris.retrieve(
        framework = framework_type,
        region = "us-east-1",
        version = framework_version,
        py_version = "py3",
        instance_type = instance_type,
        image_scope = "inference"
    )
    return image_uri

tf_image = retrieve_image("tensorflow","ml.m5.xlarge","2.3.0")


def check_model_data():
    status = subprocess.call(['test','-e',"model.tar.gz"])
    if status != 0:
        raise ValueError("Model.tar.gz failed to build succesfully")
    print("Built model.tar.gz")
    return "model.tar.gz"

#To-do: figure out how to list output files with subprocess
def build_model_package(image_uri, inference_script=None):
    if inference_script is not None:
        createDirectory = "mkdir code"
        copyInference = f"cp {inference_script} code"
        p1 = subprocess.call(createDirectory, shell=True)
        p2 = subprocess.call(copyInference, shell=True)
        createZip = "tar -cvpzf model.tar.gz ./0000001 ./code"
        p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
        output, error = p3.communicate()
        print("Created a model.tar.gz with a custom inference script.")
    else:
        createZip = "tar -cvpzf model.tar.gz ./0000001"
        p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
        output, error = p3.communicate()
        print("Created a model.tar.gz without a custom inference script.")
    return p3

model_artifact = build_model_package(tf_image, "inference.py")
print(model_artifact)




"""
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


def create_model(image_uri, model_data, inference_script=None):
    return None

def create_epc():
    return None

def create_ep():
    return None
"""