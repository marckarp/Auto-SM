import os
def check_model_artifact(framework_type, model_data):
    if framework_type == "sklearn":
        if "joblib" not in model_data:
            return False
        return True

    elif framework_type == "tensorflow":
        tf_files = ['variables', 'keras_metadata.pb', 'saved_model.pb']
        if os.path.isdir(model_data):
            files = os.listdir(model_data)
            if all(elem in files for elem in tf_files):
                return True
            return False
        return False

print(check_model_artifact('tensorflow','0000001'))