import os 
import sys
sys.path.insert(0, '..')
import io
import joblib
import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError


ACCESS_KEY = str(os.getenv("AWS_ACCESS_KEY_ID"))
SECRET_KEY = str(os.getenv("AWS_SECRET_ACCESS_KEY"))

# user function;
def read_s3_file(bucket_name, key_value):
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)
        obj = s3.get_object(Bucket=bucket_name, Key=key_value)
        return io.BytesIO(obj['Body'].read())
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            print("Key doesn't match. Please check the key value entered.")


def upload_to_aws(local_file, bucket, s3_file):
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False


if __name__ == "__main__":
    df = read_s3_file('bucket1-aws-2023', 'xtrain')
    print(df)