import sagemaker
region = "us-east-1"
# retrieve sklearn image
image_uri = sagemaker.image_uris.retrieve(
    framework="pytorch",
    region=region,
    version="1.8",
    py_version="py3",
    instance_type="ml.m5.xlarge",
    image_scope="inference"
)
print(image_uri)