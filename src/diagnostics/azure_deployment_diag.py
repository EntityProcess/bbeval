#!/usr/bin/env python3
"""
Diagnostic script to check Azure OpenAI deployment availability.
This is not a unit test; it's a manual troubleshooting utility.
"""

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables
load_dotenv()

# Get Azure OpenAI configuration
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-5-chat")

print(f"Endpoint: {endpoint}")
print(f"API Key: {'***' + api_key[-4:] if api_key else 'None'}")
print(f"Model/Deployment: {model}")

# Test common deployment names
deployment_names = [
    "gpt-5-chat",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4o",
    "gpt-4.1"
]

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-02-01"
)

print("\nTesting deployment names:")
for deployment in deployment_names:
    try:
        print(f"Testing {deployment}...", end=" ")
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "user", "content": "Hello, world!"}
            ],
            max_tokens=10
        )
        print("✅ SUCCESS")
        print(f"  Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")

print("\nIf none of these work, you'll need to:")
print("1. Check your Azure OpenAI resource in the Azure portal")
print("2. Look at the 'Deployments' section")
print("3. Use the exact deployment name shown there")
print("4. Or create a new deployment if none exist")
