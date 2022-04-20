from json.tool import main
from automodel import frameworks as fwk

if __name__ == "__main__":
    ''' '''
    sk_model = fwk.SKLearnModel(version = "0.23-1", model_data = 'sklearn/model')
    tf_model = fwk.TensorflowModel(version = "1.15.2", model_data = 'tensorflow/model')
    
    sk_model.deploy_to_sagemaker()
    # tf_model.deploy_to_sagemaker()