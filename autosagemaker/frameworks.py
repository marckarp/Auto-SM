import os
import subprocess
from autosagemaker.auto_sagemaker import AutoSageMaker
import sagemaker
from time import gmtime, strftime
import boto3


class SKLearnModel(AutoSageMaker):
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


class TensorFlowModel(AutoSageMaker):
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


class PyTorchModel(AutoSageMaker):
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