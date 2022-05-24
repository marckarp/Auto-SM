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
endpoint_name = "auto-sm-endpoint-2022-05-05-20-38-04"
jsons = JSONSerializer()
payload = jsons.serialize(sampInput)
print(payload)
response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload)
result = json.loads(response['Body'].read().decode())['outputs']
print(result)