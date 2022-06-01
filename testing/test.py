from autosagemaker import frameworks as fwk
import os

if __name__ == "__main__":
    ''' '''
    sk_model = fwk.SKLearnModel(
        version = "0.23-1", 
        model_data = 'model.joblib',
        inference_option = 'real-time',
        inference = 'inference.py'
    )
    #tf_model = fwk.TensorflowModel(version = "1.15.2", model_data = 'tensorflow/model')
    
    sk_model.deploy_to_sagemaker()
    # tf_model.deploy_to_sagemaker()