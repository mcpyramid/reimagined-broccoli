import boto3
import re
import os
import wget
from time import gmtime, strftime
import time
import sys
import json
from sagemaker.sklearn.estimator import SKLearnModel
import sagemaker
from pprint import pprint

start = time.time()
configuration_file = sys.argv[1]

with open(configuration_file) as f:
    data = json.load(f)

model_data = data["Parameters"]["ModelData"]
role =  data["Parameters"]["SageMakerRole"]
entry_point = data["Parameters"]["EntryPoint"]
source_dir = data["Parameters"]["SourceDirectory"]
model_name = data["Parameters"]["ModelName"]
environment = data["Parameters"]["Environment"]
image = '118104210923.dkr.ecr.us-east-1.amazonaws.com/scikit-nlp'
print("=== Creating SK Learn Model given the following parameters ===")
pprint(data)
sklearn_model = SKLearnModel(model_data = model_data, role = role,
  entry_point = entry_point, source_dir = source_dir, name = environment + "-" + model_name, image=image)
predictor = sklearn_model.deploy(instance_type = "ml.c4.xlarge", initial_instance_count = 1)
print("=== Done deploying. Endpoint Name: [{}] ===".format(sklearn_model.name))
end = time.time()
print(end - start)