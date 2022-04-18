import os
from .auto_model import AutoModel

class SKLearnModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'sklearn'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", os.listdir(self._model_data_))
        files = os.listdir(self._model_data_)
        models = []
        for f in files:
            if f.split('.')[-1] == 'joblib':
                models.append(f)
        if len(models) == 0:
            raise ValueError("Sklearn model artifact must be of type model.joblib for SageMaker.")
        return models

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
        print("Model data contains: ", os.listdir(self._model_data_))
        files = os.listdir(self._model_data_)
        models = []
        for f in files:
            if f.split('.')[-1] in ['.pt', '.pth']:
                models.append(f)
        if len(models) == 0:
            raise ValueError("PyTorch model artifact must be of type model.pt or model.pth for SageMaker.")
        return models

    def package(self):
        self._check_artifact_()
        return super().package()

class TensorflowModel(AutoModel):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'tensorflow'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", os.listdir(self._model_data_))
        tf_files = ['assets', 'variables', 'keras_metadata.pb', 'saved_model.pb']
        files = os.listdir(self._model_data_)
        models = []
        if len(files):
            for elem in files:
                if elem in tf_files:
                    models.append(elem)
        
        if not len(models) == len(tf_files):
            raise ValueError("Tensorfflow model folder must container ['assets', 'variables', 'keras_metadata.pb', 'saved_model.pb'] for SageMaker.")
        return models

    def package(self):
        self._check_artifact_()
        return super().package()

if __name__ == "__main__":
    ''' '''
    sklearn_model = SKLearnModel(version = '0.23-1')
    sklearn_model.deploy_to_sagemaker()