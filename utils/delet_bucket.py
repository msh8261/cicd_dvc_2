import os
import boto3

ACCESS_KEY = str(os.getenv("AWS_ACCESS_KEY_ID"))
SECRET_KEY = str(os.getenv("AWS_SECRET_ACCESS_KEY"))

s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)
response = client.delete_bucket(
    Bucket='bucket1-aws-2023',
)

print(response)








