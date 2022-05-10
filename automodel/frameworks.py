import os
import subprocess
import sagemaker
from time import gmtime, strftime
import boto3

client = boto3.client(service_name="sagemaker")
runtime = boto3.client(service_name="sagemaker-runtime")
boto_session = boto3.session.Session()
s3 = boto_session.resource('s3')
region = boto_session.region_name
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::474422712127:role/sagemaker-role-BYOC" #add functionality to add role

class AutoModel():

    """ """
    def __init__(self, **kwargs) -> None:
        ''' '''
        self._framework_ = kwargs['framework']
        self._version_ = kwargs['version']

        self._instance_type_ = kwargs.get('instance_type', 'ml.m5.xlarge')
        self._instance_count_ = kwargs.get('instance_count', 1)
        self._model_data_ = kwargs.get('model_data', None)
        self._requirements_ = kwargs.get('requirements', None)
        self._inference_ = kwargs.get('inference', None)
        
        if not (self._requirements_ is None):
            assert os.path.isfile(self._requirements_), "Requirements must point to a valid file"

        if not (self._inference_ is None):
            assert self._inference_.split('/')[-1] == 'inference.py', "Inference script must be named inference.py"
    
        if not (self._model_data_ is None):
            #need to add logic to distinguish sklearn, tf, pytorch here
            #sklearn is just a file and script, tf is a folder
            assert not (len(self._model_data_) == 0), "Make sure to provide a model file"


    def package(self):
        ''' '''
        filename = 'model.tar.gz'
        try:
            #print(os.listdir(self._model_data_))
            print("-----In packaging function-----------")
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
        default_bucket = sagemaker_session.default_bucket()
        model_artifacts = f"s3://{default_bucket}/model.tar.gz"
        response = s3.meta.client.upload_file(filename, default_bucket, 'model.tar.gz')
        return model_artifacts

    def create_model(self, model_artifacts):
        ''' '''
        image_uri = sagemaker.image_uris.retrieve(
            framework       = self._framework_,
            region          = "us-east-1",
            version         = self._version_,
            py_version      = "py3",
            instance_type   = self._instance_type_,
            image_scope = "inference"
        )
        model_name = "auto-sm-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_model_response = client.create_model(
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
            ExecutionRoleArn = role,
        )
        return model_name

    def create_endpoint_config(self, model_name):
        ''' '''
        endpoint_config = "auto-sm-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        endpoint_config_response = client.create_endpoint_config(
            EndpointConfigName=endpoint_config,
            ProductionVariants=[
                {
                    "VariantName": "sklearnvariant",
                    "ModelName": model_name,
                    "InstanceType": "ml.c5.large",
                    "InitialInstanceCount": 1
                },
            ],
        )
        return endpoint_config

    def create_endpoint(self, endpoint_config_name):
        ''' '''
        endpoint_name = "auto-sm-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        create_endpoint_response = client.create_endpoint(
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


class SKLearnModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'sklearn'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
        #files = os.listdir(self._model_data_)
        print("Checking model artifact for sklearn------------------")
        #print(files)
        print("-------------------")
        if "joblib" not in self._model_data_:
            raise ValueError("Your sklearn model artifact needs to be in joblib format")
        return True

    def package(self):
        self._check_artifact_()
        return super().package()


class TensorFlowModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'tensorflow'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
        print("Checking model artifact for tensorflow------------------")
        print("-------------------")

        tf_files = ['variables', 'keras_metadata.pb', 'saved_model.pb']
        if os.path.isdir(self._model_data_):
            files = os.listdir(self._model_data_)
            print(files)
            if all(elem in files for elem in tf_files):
                return True
            return False
        return False

    def package(self):
        self._check_artifact_()
        return super().package()


class PyTorchModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'pytorch'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
        #files = os.listdir(self._model_data_)
        print("Checking model artifact for sklearn------------------")
        #print(files)
        print("-------------------")
        if "pth" not in self._model_data_:
            raise ValueError("Your pytorch model artifact needs to be in .pth format")
        return True

    def package(self):
        self._check_artifact_()
        return super().package()


if __name__ == "__main__":
    ''' '''

    #To-do: Need to find a way to package when there is no inference script


    #Edge Case 1: Sklearn with inference script (working)
    #To-do: Kirit try with a different sklearn model + inference script
    #sklearn_model = SKLearnModel(version = '0.23-1', model_data = 'model.joblib', inference="inference.py")
    #sklearn_model.deploy_to_sagemaker()


    #Edge Case 2: Sklearn without inference script (broken/bug)


    #Edge Case 3: TF with inference script (working)
    ##To-do: Kirit try with a different TF model + inference script
    #tensorflow_model = TensorFlowModel(version = '2.3.0', model_data = '0000001', inference='inference.py')
    #tensorflow_model.deploy_to_sagemaker()


    #Edge Case 4: TF without inference script (this works too, just didn't test here)


    #Edge Case 5: PT with inference script (working)
    #To-do: Ram try with a PT model + inference script
    pytorch_model = PyTorchModel(version = '1.8', model_data = 'model.pth', inference='inference.py')
    pytorch_model.deploy_to_sagemaker()


    #Edge Case 6 PT without inference script