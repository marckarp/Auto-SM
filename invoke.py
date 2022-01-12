import boto3
import json
from sagemaker.serializers import JSONSerializer
import pandas as pd

#Grab sample test data
test = pd.read_csv('Data/Boston/train.csv')
test = test[:1]
testX = test.drop("TARGET", axis=1)
testX = testX[:1].values.tolist()
sampInput = {"inputs": testX}
sampInput
print(sampInput)



runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
endpoint_name = "tf-ep-autosm2022-01-12-20-30-28"
jsons = JSONSerializer()
payload = jsons.serialize(sampInput)
response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload)
result = json.loads(response['Body'].read().decode())['outputs']
print(result)