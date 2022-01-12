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


#Setup (need to automate this, sts)
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


def check_model_package():
    list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    model_data = 'model.tar.gz'
    if model_data not in list_of_files.stdout:
        #Edit this to a custom exception later
        raise ValueError("model.tar.gz file was not created, make sure to have provided an image_uri")
    return True

def custom_inference_package(model_data, inference_script):
    createDirectory = "mkdir code"
    copyInference = f"cp {inference_script} code"
    p1 = subprocess.call(createDirectory, shell=True)
    p2 = subprocess.call(copyInference, shell=True)
    createZip = f"tar -cvpzf model.tar.gz ./{model_data} ./code"
    p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
    output, error = p3.communicate()

def inference_package(model_data):
    createZip = f"tar -cvpzf model.tar.gz ./{model_data}"
    p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
    output, error = p3.communicate()


def build_model_package(model_data, inference_script=None):
    if model_data is None:
        raise ValueError("You need to provide the file path for the directory with your model data.")
    if inference_script is not None:
        custom_inference_package(model_data, inference_script)
    else:
        inference_package(model_data)
    if check_model_package:
        default_bucket = sagemaker_session.default_bucket()
        model_artifacts = f"s3://{default_bucket}/model.tar.gz"
        response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')
    return model_artifacts

model_data = build_model_package('0000001', "inference.py")


def create_model(image_uri, model_data):
    #Step 1: Model Creation
    model_name = "tf-autosm" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    #print("Model name: " + model_name)
    create_model_response = client.create_model(
        ModelName=model_name,
        Containers=[
            {
                "Image": image_uri,
                "Mode": "SingleModel",
                "ModelDataUrl": model_data,
                "Environment": {'SAGEMAKER_SUBMIT_DIRECTORY': model_data,
                                'SAGEMAKER_PROGRAM': 'inference.py'} 
            }
        ],
        ExecutionRoleArn=role,
    )
    model_arn = create_model_response["ModelArn"]
    return model_name, model_arn

model = create_model(tf_image, model_data)
model_name, model_arn = model[0], model[1]
print("Model Name: " + model_name)
print("Model arn: " + model_arn)

#have to adjust to paramterize instance type later
def create_epc(model_name):
    epc_name = "tf-epc-autosm" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    endpoint_config_response = client.create_endpoint_config(
        EndpointConfigName=epc_name,
        ProductionVariants=[
            {
                "VariantName": "tfvariant",
                "ModelName": model_name,
                "InstanceType": "ml.c5.xlarge",
                "InitialInstanceCount": 1
            },
        ],
    )
    epc_arn = endpoint_config_response["EndpointConfigArn"]
    return epc_name, epc_arn

epc = create_epc(model_name)
epc_name, epc_arn = epc[0], epc[1]
print("EPC Name: " + epc_name)
print("EPC arn: " + epc_arn)

def monitor_ep(ep_name):
    describe_endpoint_response = client.describe_endpoint(EndpointName=ep_name)
    while describe_endpoint_response["EndpointStatus"] == "Creating":
        describe_endpoint_response = client.describe_endpoint(EndpointName=ep_name)
        print(describe_endpoint_response["EndpointStatus"])
        time.sleep(15)
    return describe_endpoint_response

def create_ep(epc_name):
    ep_name = "tf-ep-autosm" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    create_endpoint_response = client.create_endpoint(
        EndpointName=ep_name,
        EndpointConfigName=epc_name,
    )
    ep_arn = create_endpoint_response["EndpointArn"]
    print(monitor_ep(ep_name))
    return ep_name, ep_arn

ep_status = create_ep(epc_name)
ep_name, ep_arn = ep_status[0], ep_status[1]
print("Endpoint name: " + ep_name)
print("Endpoint arn: " + ep_arn)