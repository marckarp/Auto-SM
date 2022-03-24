from json.tool import main
import os

class Model():
    """ """
    def __init__(self, **kwargs) -> None:
        ''' '''
        self._framework_ = ''
        self._version_ = ''
        self._model_file_ = kwargs.get('model_file', None)
        self._requirements_ = kwargs.get('requirements', None)
        if not (self._requirements_ is None):
            assert os.path.isfile(self._requirements_), "Requirements must point to a valid file"

    def compile(self):
        ''' '''
        pass

    def push_s3(self):
        ''' '''
        pass

    def deploy(self):
        ''' '''
        pass

    @property
    def Framework(self):
        ''' '''
        return (self._framework_, self._version_)

class VertexModel(Model):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
class DatabricksModel(Model):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class DataikuModel(Model):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

class KubernetesModel(Model):
    """ """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
if __name__ == '__main__':
    ''' '''
    pass
