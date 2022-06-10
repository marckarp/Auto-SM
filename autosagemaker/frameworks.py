import os
from autosagemaker.auto_sagemaker import AutoSageMaker


class SKLearnModel(AutoSageMaker):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'sklearn'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        
        ## Retrieve list of versions from SM
        ## Check if version in list of versions

        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
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
        
        ## Retrieve list of versions from SM
        ## Check if version in list of versions

        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
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
    
class KerasModel(AutoSageMaker):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'tensorflow'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"

        ## Retrieve list of versions from SM
        ## Check if version in list of versions

        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        import os
        import tensorflow as tf
        import tensorflow.keras as keras
        from keras.models import model_from_json
        from keras import backend as K
        from keras.models import load_model
        
        print("Model data contains: ", self._model_data_)
        if os.path.isdir(self._model_data_):
            files = os.listdir(self._model_data_)
            print(files)
            json_file =  [file for file in files if ".json" in file]
            h5_file =  [file for file in files if ".h5" in file]
           
            if json_file !=[] and h5_file!= []:
                print(".json and .h5 files found")
                self.json_filename = json_file[0]
                self.h5_filename = h5_file[0]
                with open(os.path.join(self._model_data_, json_file[0]), 'r') as fp:
                    loaded_model_json = fp.read()
                self.loaded_model = model_from_json(loaded_model_json)
                self.loaded_model.load_weights(os.path.join(self._model_data_, h5_file[0]))
                return True
            
            elif h5_file!= []:
                print("Single .h5 files found")
                self.loaded_model = keras.models.load_model(os.path.join(self._model_data_, h5_file[0]))
                return True
            else:
                print("Please provide Keras model artifacts")
            return False
        return False
    
    def _create_saved_model_format(self):
        from tensorflow.python.saved_model import builder
        from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
        from tensorflow.python.saved_model import tag_constants
        from keras import backend as K
        
        # Note: This directory structure will need to be followed 
        model_version = '0000001'
        export_dir = os.path.join(self._model_data_, '0000001')
        # Build the Protocol Buffer SavedModel at path defined by export_dir variable
        builder = builder.SavedModelBuilder(export_dir)

        # Create prediction signature to be used by TensorFlow Serving Predict API
        signature = predict_signature_def(
            inputs={"inputs": self.loaded_model.input}, outputs={"score": self.loaded_model.output})

        # Save the meta graph and variables
        builder.add_meta_graph_and_variables(
            sess=K.get_session(), tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
        builder.save()

    def package(self):
        
        assert self._check_artifact_()
        self._create_saved_model_format()
        return super().package()


class PyTorchModel(AutoSageMaker):
    """ """
    def __init__(self, **kwargs) -> None:
        kwargs['framework'] = 'pytorch'
        assert 'version' in kwargs, "Framework version must be specified."
        assert 'model_data' in kwargs, "Folder with model data must be provided"
        
        ## Retrieve list of versions from SM
        ## Check if version in list of versions

        super().__init__(**kwargs)

    def _check_artifact_(self):
        ''' '''
        print("Model data contains: ", self._model_data_)
        if "pth" not in self._model_data_:
            raise ValueError("Your pytorch model artifact needs to be in .pth format")
        return True

    def package(self):
        self._check_artifact_()
        return super().package()


if __name__ == "__main__":
    ''' '''
