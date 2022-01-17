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


def check_model_package():
    list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    model_data = 'model.tar.gz'
    if model_data not in list_of_files.stdout:
        #Edit this to a custom exception later
        raise ValueError("model.tar.gz file was not created, make sure to have provided an image_uri")
    return True


"""
Build a function to automate zip file creation, take out redundance of next function
"""

def build_zip_file(zip_file):
    p3 = subprocess.Popen(zip_file.split(), stdout=subprocess.PIPE)
    output, error = p3.communicate()


def custom_inference_package(framework_type, model_data, inference_script):
    #works for tensorflow right now not sklearn/pytorch
    print("In custom inference package")
    print(framework_type)
    if framework_type == "tensorflow":
        print("Building a custom tf package")
        createDirectory = "mkdir code"
        copyInference = f"cp {inference_script} code"
        p1 = subprocess.call(createDirectory, shell=True)
        p2 = subprocess.call(copyInference, shell=True)
        zip_file = f"tar -cvpzf model.tar.gz ./{model_data} ./code"
        build_zip_file(zip_file)

    elif framework_type == "sklearn":
        print("Building a custom sklearn package")
        zip_file = f"tar -cvpzf model.tar.gz {model_data} {inference_script}"
        build_zip_file(zip_file)


def inference_package(framework_type, model_data):
    print("In inference package function")
    if framework_type == "tensorflow":
        zip_file = f"tar -cvpzf model.tar.gz ./{model_data}"
        build_zip_file(zip_file)
    elif framework_type == "sklearn":
        print("building sklearn pakcage without inference script")
        zip_file = f"tar -cvpzf model.tar.gz {model_data}"
        build_zip_file(zip_file)


def build_model_package(framework_type, model_data, inference_script=None):
    """Need to edit and add functionality to take in sklearn/pytorch, adjust bash commands in custom_inference_package
    and inference_package modules to deploy endpoint properly"""
    print("Building model package")
    print(framework_type)
    if framework_type is None:
        raise ValueError("You need to provide the framework type that you are working with.")
    if model_data is None:
        raise ValueError("You need to provide the file path for the directory with your model data.")
    if inference_script is not None:
        print("Building package with a custom inference script")
        custom_inference_package(framework_type, model_data, inference_script)
    else:
        print("Building package without an inference script")
        inference_package(framework_type, model_data)
    if check_model_package:
        default_bucket = sagemaker_session.default_bucket()
        model_artifacts = f"s3://{default_bucket}/model.tar.gz"
        response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')
    return model_artifacts


def create_model(image_uri, model_data):
    """Builds a SageMaker Model Entity, which will be used to create an Endpoint Configuration.

    Args:
        image_uri (str): The image that has been retrieved for your specific framework.
        model_data (str): The S3 URI of the model data.

    Returns:
        tuple: Contains SageMaker model name as first item and model arn as second item.
    """
    model_name = "tf-autosm" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
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


#have to adjust to paramterize instance type later
def create_epc(model_name):
    """Creates SageMaker Endpoint Configuration.

    Args:
        model_name (str): SageMaker Model that was created in create_model function.

    Returns:
        tuple: Contains SageMaker Endpoint Configuration name as first item and Endpoint Configuration arn as second item.
    """
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


def monitor_ep(ep_name):
    """Monitors endpoint creation, the process should take 3-6 minutes.

    Args:
        ep_name (str): Endpoint that is being created.

    Returns:
        str: Returns endpoint status "creating" till it is "InService".
    """
    describe_endpoint_response = client.describe_endpoint(EndpointName=ep_name)
    while describe_endpoint_response["EndpointStatus"] == "Creating":
        describe_endpoint_response = client.describe_endpoint(EndpointName=ep_name)
        print(describe_endpoint_response["EndpointStatus"])
        time.sleep(15)
    return describe_endpoint_response

def create_ep(epc_name):
    """Creates endpoint, will use monitor_ep to monitor endpoint creation.

    Args:
        epc_name (str): Using endpoint configuration created by create_epc call.

    Returns:
        tuple: Contains SageMaker Endpoint name as first item and Endpoint arn as second item.
    """
    ep_name = "tf-ep-autosm" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    create_endpoint_response = client.create_endpoint(
        EndpointName=ep_name,
        EndpointConfigName=epc_name,
    )
    ep_arn = create_endpoint_response["EndpointArn"]
    print(monitor_ep(ep_name))
    return ep_name, ep_arn


def check_model_artifact(framework_type, model_data):
    if framework_type == "sklearn":
        if "joblib" not in model_data:
            return False
        return True

    elif framework_type == "tensorflow":
        tf_files = ['assets', 'variables', 'keras_metadata.pb', 'saved_model.pb']
        if os.path.isdir(model_data):
            files = os.listdir(model_data)
            if all(elem in files for elem in tf_files):
                return True
            return False
        return False