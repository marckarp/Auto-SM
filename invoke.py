#TF Invocation

"""
import boto3
import json
from sagemaker.serializers import JSONSerializer
import pandas as pd

#Grab sample test data
test = pd.read_csv('Data/Boston/train.csv')
test = test[:3]
testX = test.drop("TARGET", axis=1)
testX = testX[:3].values.tolist()
sampInput = {"inputs": testX}
sampInput
print(sampInput)



runtime_sm_client = boto3.client(service_name='sagemaker-runtime')
endpoint_name = "tf-ep-autosm2022-01-16-21-46-05"
jsons = JSONSerializer()
payload = jsons.serialize(sampInput)
response = runtime_sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=payload)
result = json.loads(response['Body'].read().decode())['outputs']
print(result)
"""

#Sklearn Invocation
"""

import boto3
import json

runtime_client = boto3.client('sagemaker-runtime')
content_type = "application/json"
request_body = {"Input": [[0.09178, 0.0, 3.05, 1.0, 0.51, 6.416, 84.1, 2.6463, 5.0, 296.0, 16.6, 395.5, 9.04]]}
data = json.loads(json.dumps(request_body))
payload = json.dumps(data)
endpoint_name = "tf-ep-autosm2022-01-16-21-35-06"

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload)
result = json.loads(response['Body'].read().decode())['Output']
print(result)

"""