#for testing random code snippets during build

"""
tf_versions = ["1.10.0", "1.11.0", "1.12.0", "1.13.0", "1.14.0", "1.15.0", "1.15.2", "1.15.3", 
        "1.15.4", "1.15.5", "1.4.1", "1.5.0", "1.6.0", "1.7.0", "1.8.0", "1.9.0", "2.0.0", "2.0.1", "2.0.2",
        "2.0.3", "2.0.4", "2.1.0", "2.1.1", "2.1.2", "2.1.3", "2.2.0", "2.2.1", "2.2.2", "2.3.0", "2.3.1",
        "2.3.2", "2.4.1", "2.4.3", "2.5.1", "2.6.0", "1.10", "1.11", "1.12", "1.13", "1.14", "1.15", "1.4",
        "1.5", "1.6", "1.7", "1.8", "1.9", "2.0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6"]


from re import I
import subprocess
sampStr = "testfolder"

#bashCommand = f"mkdir {sampStr}"
#p1 = subprocess.check_output(bashCommand, shell=True)
#print(p1)
"""

"""
import subprocess
# file and directory listing
returned_text = subprocess.check_output("model.tar.gz", shell=True, universal_newlines=True)
print("dir command to list file and directory")
print(returned_text)
"""

"""
#check status or existence of a file
list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
print(list_of_files.stdout)
if 'query.py' in list_of_files.stdout:
        print("found")
else:
        print("not created")
#ls_lines = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE).stdout.splitlines()
#print(ls_lines)
#print(check_model_data)
#if "model.tar.gz" in check_model_data:
 #       print("found")
#rint("not found")


list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
model_artifact = 'model.tar.gz'
if model_artifact not in list_of_files.stdout:
        raise ValueError("model.tar.gz file was not created")
print("Found model.tar.gz returning model artifact")
"""

"""
cleaned
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


def check_model_package():
    list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    model_data = 'model.tar.gz'
    if model_data not in list_of_files.stdout:
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
"""

"""
def check_model_data():
    status = subprocess.call(['test','-e',"model.tar.gz"])
    if status != 0:
        raise ValueError("Model.tar.gz failed to build succesfully")
    #print("Built model.tar.gz")
    return "model.tar.gz"
"""

"""
#To-do: clean this function
def build_model_package(image_uri, inference_script=None):
    if inference_script is not None:
        createDirectory = "mkdir code"
        copyInference = f"cp {inference_script} code"
        p1 = subprocess.call(createDirectory, shell=True)
        p2 = subprocess.call(copyInference, shell=True)
        createZip = "tar -cvpzf model.tar.gz ./0000001 ./code"
        p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
        output, error = p3.communicate()
        #print("Created a model.tar.gz with a custom inference script.")
    else:
        createZip = "tar -cvpzf model.tar.gz ./0000001"
        p3 = subprocess.Popen(createZip.split(), stdout=subprocess.PIPE)
        output, error = p3.communicate()
        #print("Created a model.tar.gz without a custom inference script.")
    
    list_of_files = subprocess.run(['ls', '-l'], capture_output=True, text=True)
    model_data = 'model.tar.gz'
    if model_data not in list_of_files.stdout:
        raise ValueError("model.tar.gz file was not created, make sure to have provided an image_uri")
    #print("Found model.tar.gz returning model artifact")
    default_bucket = sagemaker_session.default_bucket()
    #print(default_bucket)
    model_artifacts = f"s3://{default_bucket}/model.tar.gz"
    response = s3.meta.client.upload_file('model.tar.gz', default_bucket, 'model.tar.gz')
    return model_artifacts
"""

import os

def check_model_artifact(framework_type, model_data):
    if framework_type == "sklearn":
        if "joblib" not in model_data:
            raise ValueError("For the sklearn framework your model data must be saved using the joblib module. \n"
            "Check out the following link for an example of using the joblib package: https://scikit-learn.org/0.18/modules/model_persistence.html")


if os.path.exists("sadasa") is False:
    raise ValueError("Cannot find your inference script, please enter the proper path to your code.")


















"""
def custom_inference_package(model_data, inference_script):
    #works for tensorflow right now not sklearn/pytorch
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
    Need to edit and add functionality to take in sklearn/pytorch, adjust bash commands in custom_inference_package
    and inference_package modules to deploy endpoint properly
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
"""