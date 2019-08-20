import boto3
import re
import os
import wget
from time import gmtime, strftime
import time
import sys
import json
from sagemaker.sklearn.estimator import SKLearn
from pprint import pprint
import sagemaker
start = time.time()

role = sys.argv[1]
bucket = sys.argv[2]
stack_name = sys.argv[3]
commit_id = sys.argv[4]
commit_id = commit_id[0: 7]
training_image = '118104210923.dkr.ecr.us-east-1.amazonaws.com/scikit-nlp'
timestamp = time.strftime('%Y-%m-%d-%H-%M-%S', time.gmtime())
cwd = os.getcwd()
print(cwd)
import tarfile
print("Creating tar file")
def make_tarfile(output_filename, source_dir):
  with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(source_dir, arcname = '')

tarFileName = stack_name + "-" + commit_id + "-" + timestamp + '.tar.gz'
make_tarfile(tarFileName, cwd + '/Scikit-Learn/code')

def upload_to_s3(channel, file):
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    print("Uploading to s3://{}//{}".format(bucket, key))
    s3.Bucket(bucket).put_object(Key = key, Body = data)
    return 's3://{}/{}'.format(bucket,key)

source_dir = upload_to_s3('source', tarFileName)

train_input = 's3://{}/data/train'.format(bucket)
job_name = stack_name + "-" + commit_id
hyperparameters = {
  "sagemaker_container_log_level": str(20),
  "sagemaker_enable_cloudwatch_metrics": "false",
  "sagemaker_job_name": "\"{}\"".format(job_name),
  "sagemaker_program": "\"{}\"".format("train.py"),
  "sagemaker_region": "\"{}\"".format("us-east-1")
}

sklearn = SKLearn(
  base_job_name = job_name,
  image_name = '118104210923.dkr.ecr.us-east-1.amazonaws.com/scikit-nlp',
  entry_point = 'train.py',
  source_dir = source_dir,
  train_instance_type = "ml.m5.24xlarge",
  output_path = 's3://mctestraaa-pipeline-data/model/',
  hyperparameters = hyperparameters,
  role = role)
sklearn.fit({
  'train': train_input
})

model_params = sklearn.create_model()


config_data_qa = {
  "Parameters":
    {
        "Environment": "qa",
        "ModelData": model_params.model_data,
        "ModelName": model_params.name,
        "SageMakerRole": model_params.role,
        "StackName": stack_name,
        "SourceDirectory": model_params.source_dir
    }
}

config_data_prod = {
  "Parameters":
    {
        "Environment": "prod",
        "ModelData": model_params.model_data,
        "ModelName": model_params.name,
        "SageMakerRole": model_params.role,
        "StackName": stack_name,
        "SourceDirectory": model_params.source_dir
    }
}

print("=== QA CONFIG ===")
pprint(config_data_qa)

print("=== PROD CONFIG ===")
pprint(config_data_prod)

json_config_data_qa = json.dumps(config_data_qa)
json_config_data_prod = json.dumps(config_data_prod)

f = open( './configuration_qa.json', 'w' )
f.write(json_config_data_qa)
f.close()

f = open( './configuration_prod.json', 'w' )
f.write(json_config_data_prod)
f.close()

end = time.time()
print(end - start)