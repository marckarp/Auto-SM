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
    
    elif framework_type == "pytorch":
        if ".pth" not in model_data:
            return False
        return True