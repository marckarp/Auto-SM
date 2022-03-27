from json.tool import main
from .auto_model import AutoModel

class SKLearnModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'sklearn'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'role' in kwargs, "Role must be provided"
        super().__init__(**kwargs)

    def package(self):
        return super().package()
    
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