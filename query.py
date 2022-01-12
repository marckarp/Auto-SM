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

from config import framework_types, tf_versions, pytorch_versions, sklearn_versions

class autoSM:

    def __init__(self, framework_type, instance_type, framework_version, model_data, inference_script=None):
        self.framework_type = framework_type
        self.instance_type = instance_type
        self.framework_version = framework_version
        self.model_data = model_data
        self.inference_script = inference_script if inference_script is not None else ""

        if self.model_data is None:
            raise ValueError("Please make sure to enter the path for your model data/artifacts.")

        if self.framework_type not in framework_types:
            raise ValueError(f"Enter one of the following frameworks that is supported: {framework_types}")
        
        if self.framework_type == "sklearn":
            if self.framework_version not in sklearn_versions:
                raise ValueError(f"Unsupported sklearn version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer sklearn versions.\n"
                f"Supported sklearn versions: {sklearn_versions}")
        
        elif self.framework_type == "tensorflow":
            if self.framework_version not in tf_versions:
                raise ValueError(f"Unsupported tf version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer Tensorflow versions.\n"
                f"Supported TensorFlow versions: {tf_versions}")
                
        elif self.framework_type == "pytorch":
            if self.framework_version not in pytorch_versions:
                raise ValueError(f"Unsupported pytorch version. You may need to upgrade your SDK version (pip install -U sagemaker) for newer PyTorch versions.\n"
                f"Supported PyTorch versions: {pytorch_versions}")
    
    def describe_job(self):
        print(f"Framework Type: {self.framework_type}")
        print(f"Instance Type: {self.instance_type}")
        print(f"Framework Version: {self.framework_version}")

if __name__ == '__main__':
    auto_model = autoSM("pytorch", "ml.c5.xlarge", "1.9.0", None)
    auto_model.describe_job()