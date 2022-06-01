import os
import subprocess
import sagemaker
from time import gmtime, strftime
from .sm_client import AutoSMClient

class AutoSageMaker():

    """ """
    def __init__(self, **kwargs) -> None:
        ''' '''
        self._framework_ = kwargs['framework']
        self._version_ = kwargs['version']

        self._auto_sm_client_ = AutoSMClient(**kwargs)

        self._instance_type_ = kwargs.get('instance_type', 'ml.m5.xlarge')
        self._instance_count_ = kwargs.get('instance_count', 1)

        self._memory_size_ = kwargs.get('memory_size', 4096)
        self._concurrency_ = kwargs.get('concurrency', 10)

        self._model_data_ = kwargs.get('model_data', None)
        self._inference_option_ = kwargs.get('inference_option', 'real-time')
        self._requirements_ = kwargs.get('requirements', None)
        self._inference_ = kwargs.get('inference', None)
        
        if not (self._requirements_ is None):
            assert os.path.isfile(self._requirements_), "Requirements must point to a valid file"

        if not (self._inference_ is None):
            assert self._inference_.split('/')[-1] == 'inference.py', "Inference script must be named inference.py"
    
        if not (self._model_data_ is None):
            assert not (len(self._model_data_) == 0), "Make sure to provide a model file"

        self._sm_client_ = self._auto_sm_client_.AutoSagemakerClient
        self._role_ = self._auto_sm_client_.Role


    def package(self):
        ''' '''
        filename = 'model.tar.gz'
        try:
            print(self._model_data_)
            print(self._inference_)
            zip_file = f"tar -cvpzf model.tar.gz {self._model_data_} {self._inference_}"
            p3 = subprocess.Popen(zip_file.split(), stdout=subprocess.PIPE)
            output, error = p3.communicate()
        except:
            print("Unable to package model folder into tarball")
        return filename

    def push_s3(self, filename):
        ''' '''
        s3_client = self._auto_sm_client_.AutoS3Client
        default_bucket = self._auto_sm_client_.DefaultBucket
        prefix = 'autosagemaker/{strftime("%Y-%m-%d-%H-%M-%S", gmtime())}'
        model_artifacts = f"s3://{default_bucket}/{prefix}/model.tar.gz"
        response = s3_client.upload_file(filename, default_bucket, '{prefix}/model.tar.gz')
        return model_artifacts

    def create_model(self, model_artifacts):
        ''' '''
        image_uri = sagemaker.image_uris.retrieve(
            framework       = self._framework_,
            region          = self._auto_sm_client_.Region,
            version         = self._version_,
            py_version      = "py3",
            instance_type   = self._instance_type_,
            image_scope = "inference"
        )
        model_name = "auto-sm-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_model_response = self._sm_client_.create_model(
            ModelName   = model_name,
            Containers  = [
                {
                    "Image"         : image_uri,
                    "Mode"          : "SingleModel",
                    "ModelDataUrl"  : model_artifacts,
                    "Environment"   : {
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_artifacts,
                        'SAGEMAKER_PROGRAM': 'inference.py'
                    } 
                }
            ],
            ExecutionRoleArn = self._role_,
        )
        return model_name

    def create_endpoint_config(self, model_name):
        ''' '''
        endpoint_config = "auto-sm-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        if self._inference_option_ == "real-time":
            endpoint_config_response = self._sm_client_.create_endpoint_config(
                EndpointConfigName=endpoint_config,
                ProductionVariants=[
                    {
                        "VariantName": "primaryvariant",
                        "ModelName": model_name,
                        "InstanceType": self._instance_type_,
                        "InitialInstanceCount": self._instance_count_
                    },
                ],
            )
        elif self._inference_option_ == "serverless":
            endpoint_config_response = self._sm_client_.create_endpoint_config(
                EndpointConfigName=endpoint_config,
                ProductionVariants=[
                    {
                        "VariantName": "primaryvariant",
                        "ModelName": model_name,
                        "ServerlessConfig": {
                            "MemorySizeInMB": self._memory_size_,
                            "MaxConcurrency": self._concurrency_,
                        },
                    },
                ],
            )

        else:
            raise ValueError("We only support serverless and real-time endpoints at the moment, enter one of these two options.")
            
        return endpoint_config

    def create_endpoint(self, endpoint_config_name):
        ''' '''
        endpoint_name = "auto-sm-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_endpoint_response = self._sm_client_.create_endpoint(
            EndpointName        = endpoint_name,
            EndpointConfigName  = endpoint_config_name,
        )
        return create_endpoint_response["EndpointArn"]

    def deploy_to_sagemaker(self):
        ''' '''
        # Package the model
        filename = self.package()
        print(filename)

        # Push model.tar.gz to S3
        print(f"Uploading {filename} to S3...")
        model_artifact = self.push_s3(filename)

        # Create model
        print("Creating model in SageMaker...")
        model_name = self.create_model(model_artifact)
        print(f"Created model: {model_name}")

        # Create endpoint config
        print("Creating endpoint config in SageMaker...")
        endpoint_config_name = self.create_endpoint_config(model_name)
        print(f"Created endpoint config: {endpoint_config_name}")

        # Create endpoint
        print("Creating endpoint in SageMaker...")
        endpoint = self.create_endpoint(endpoint_config_name)
        print(f"Created endpoint: {endpoint}")

    @property
    def Framework(self):
        ''' '''
        return (self._framework_, self._version_)
    
if __name__ == '__main__':
    ''' '''
    pass