import boto3

import json

import numpy as np

endpoint = 'hackathon-ei-endpoint'
 
runtime = boto3.Session().client('sagemaker-runtime') 
# Read image into memory
with open("plastic171.jpg", 'rb') as f:
    payload = f.read()
    
# print("type payload", type(payload))

# Send image via InvokeEndpoint API
response = runtime.invoke_endpoint(EndpointName=endpoint, ContentType='application/x-image', Body=payload)
result = json.loads(response['Body'].read().decode())
result = np.array(result)
prediction = result.argmax(axis=1)[0]
print("class predicted", prediction + 1)
