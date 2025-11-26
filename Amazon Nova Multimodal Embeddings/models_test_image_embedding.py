import boto3
import json

session = boto3.Session(profile_name="devan2", region_name="us-east-1")
bedrock_client = session.client('bedrock')

# List all embedding models you have access to
response = bedrock_client.list_foundation_models(
    byOutputModality='EMBEDDING'
)

print("Available Embedding Models:")
for model in response['modelSummaries']:
    print(f"  - {model['modelId']}")
    if 'amazon' in model['modelId'].lower() and 'embed' in model['modelId'].lower():
        print(f" ^^ This one might work for images")