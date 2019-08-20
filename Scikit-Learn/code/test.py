import boto3
import wget
import json
import numpy as np
import sys
import time

start = time.time()
configuration_file = sys.argv[1]



with open(configuration_file) as f:
    data = json.load(f)

model_data = data["Parameters"]["ModelData"]
role =  data["Parameters"]["SageMakerRole"]
model_name = data["Parameters"]["ModelName"]
environment = data["Parameters"]["Environment"]
stack_name = data["Parameters"]["StackName"]

endpoint_name = environment + "-" + stack_name + "-" + model_name

runtime = boto3.client('runtime.sagemaker') 

import json
def sendRequest(payload):
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                    ContentType='application/json', 
                                    Body=json.dumps(payload))
    result = response['Body'].read()
    # result will be in json format and convert it to ndarray
    #result = json.loads(result.decode('utf-8'))
    # the result will output the probabilities for all classes
    # find the class with maximum probability and print the class index
    print('========= RESULT =========')
    print(result)
    print('==========================\n')

payload = {'data': [{'ticker': 'AMZN', 'description': 'Amazon.com, Inc., is an American multinational technology company based in Seattle, Washington that focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is considered one of the Big Four technology companies along with Google, Apple, and Facebook' }]}
sendRequest(payload)

end = time.time()
seconds = end - start
seconds = repr(seconds)
print ("Time: " + seconds)