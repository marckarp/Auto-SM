from multiprocessing.sharedctypes import Value
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
import shlex
import logging

from config import framework_types, tf_versions, pytorch_versions, sklearn_versions
from utils import (retrieve_image, check_model_package, custom_inference_package, inference_package, 
build_model_package, create_model, create_epc, monitor_ep, create_ep, check_model_artifact, build_zip_file)

class autoSM:

    client = boto3.client(service_name="sagemaker")
    runtime = boto3.client(service_name="sagemaker-runtime")
    boto_session = boto3.session.Session()
    s3 = boto_session.resource('s3')
    region = boto_session.region_name
    sagemaker_session = sagemaker.Session()
    role = "arn:aws:iam::474422712127:role/sagemaker-role-BYOC" #add functionality to add role

    def __init__(self, framework_type, model_data, instance_type, framework_version, inference_script=None):
        self.framework_type = framework_type
        self.model_data = model_data
        self.instance_type = instance_type
        self.framework_version = framework_version
        self.inference_script = inference_script if inference_script is not None else ""

        self.model_name = None
        self.model_arn = None
        self.epc_name = None
        self.epc_arn = None
        self.ep_name = None
        self.ep_arn = None
        #self.ep_logs = None

        if self.model_data is None:
            raise ValueError("Please make sure to enter the path for your model data/artifacts.")
        
        if os.path.exists(self.model_data) is False:
            raise ValueError("Cannot find your model artifact please enter the proper path to your model data.")
        print("Found model data")

        if self.inference_script != "":
            if os.path.exists(self.inference_script) is False:
                raise ValueError("Cannot find your inference script, please enter the proper path to your code.")
        print("Found inference script")

        if self.framework_type not in framework_types:
            raise ValueError(f"Enter one of the following frameworks that is supported: {framework_types}")
        
        if self.framework_type == "sklearn":
            if self.framework_version not in sklearn_versions:
                raise ValueError(f"Unsupported sklearn version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer sklearn versions.\n"
                f"Supported sklearn versions: {sklearn_versions}")
            if check_model_artifact("sklearn", self.model_data) is False:
                raise ValueError("Make sure that your saved model is in joblib format.")

        elif self.framework_type == "tensorflow":
            if self.framework_version not in tf_versions:
                raise ValueError(f"Unsupported tf version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer Tensorflow versions.\n"
                f"Supported TensorFlow versions: {tf_versions}")
            if check_model_artifact("tensorflow", self.model_data) is False:
                raise ValueError("Make sure you saved your TensorFlow model in the following format.")
                
        elif self.framework_type == "pytorch":
            if self.framework_version not in pytorch_versions:
                raise ValueError(f"Unsupported pytorch version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer PyTorch versions.\n"
                f"Supported PyTorch versions: {pytorch_versions}")
        

    def deploy(self):
        image = retrieve_image(self.framework_type, self.instance_type, self.framework_version)
        #Edit build model package function to take in framework type
        if self.inference_script is not None:
            print("Entering utils now")
            model_data = build_model_package(self.framework_type, self.model_data, self.inference_script)
        else:
            print("Entering utils without a custom inference script")
            model_data = build_model_package(self.framework_type, self.model_data)
        model = create_model(image, model_data)
        self.model_name, self.model_arn = model[0], model[1]
        epc = create_epc(self.model_name)
        self.epc_name, self.epc_arn = epc[0], epc[1]
        ep = create_ep(self.epc_name)
        self.ep_name, self.ep_arn = ep[0], ep[1]
        ep_logs = {"Model Info ": [self.model_name, self.model_arn], "Endpoint Config Info ": [self.epc_name, self.ep_arn],
        "Endpoint Info ": [self.ep_name, self.ep_arn]}
        return ep_logs
        
    def describe_job(self):
        print(f"Framework Type: {self.framework_type}")
        print(f"Model Data Directory: {self.model_data}")
        print(f"Instance Type: {self.instance_type}")
        print(f"Framework Version: {self.framework_version}")
        print(f"Endpoint Name: {self.ep_name}")
        print(f"Endpoint Arn: {self.ep_arn}")

if __name__ == '__main__':


    #TF Example w/ custom inference script
    #auto_model = autoSM("tensorflow", '0000001', "ml.c5.xlarge", "2.3.0", "inference.py")
    #auto_model.deploy()
    #auto_model.describe_job()

    #Sklearn Example w/ custom inference script
    auto_model = autoSM(framework_type="sklearn", model_data= "model.joblib", 
    instance_type = "ml.c5.xlarge", framework_version="0.23-1", inference_script="inference.py")
    auto_model.deploy()
    auto_model.describe_job()