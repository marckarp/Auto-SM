from json.tool import main
from .auto_model import AutoModel
from framework_check import check_model_artifact

class SKLearnModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'sklearn'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'role' in kwargs, "Role must be provided"
        super().__init__(**kwargs)

    def package(self):
        return super().package()
    
    def check_artifact(self):
        
        ## Before packaging
        #check that model data is proper artifact (joblib, etc)
        #if proper artifact check for inference script
        #then create inference package
        filename = 'model.tar.gz'
        if check_model_artifact("sklearn", self._model_file_):
            self.package()
        raise ValueError("Sklearn model artifact must be of type model.joblib for SageMaker.")
    
class PyTorchModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'pytorch'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'role' in kwargs, "Role must be provided"
        super().__init__(**kwargs)

    def package(self):
        return super().package()

class TensorflowModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'tensorflow'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'role' in kwargs, "Role must be provided"
        super().__init__(**kwargs)

    def package(self):
        return super().package()

if __name__ == "__main__":
    ''' '''
    sklearn_model = SKLearnModel(version = '0.23-1')
    sklearn_model.deploy_to_sagemaker()