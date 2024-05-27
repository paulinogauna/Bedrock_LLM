#!/usr/bin/env python
# coding: utf-8

# # Lesson 1 - Your first generations with Amazon Bedrock

# Welcome to Lesson 1. 
# 
# You'll start with using Amazon Bedrock to prompt a model and customize how it generates its response.
# 
# **Note:** To access the `requirements.txt` file, go to `File` and click on `Open`. Here, you will also find all helpers functions and datasets used in each lesson.
#  
# I hope you enjoy this course!

# ### Import all needed packages

# In[ ]:


import boto3
import json


# ### Setup the Bedrock runtime

# In[ ]:


bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-west-2')


# In[ ]:


prompt = "Write a one sentence summary of Las Vegas."


# In[ ]:


kwargs = {
    "modelId": "amazon.titan-text-lite-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt
        }
    )
}


# In[ ]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[ ]:


response


# In[ ]:


response_body = json.loads(response.get('body').read())


# In[ ]:


print(json.dumps(response_body, indent=4))


# In[ ]:


print(response_body['results'][0]['outputText'])


# ### Generation Configuration

# In[ ]:


prompt = "Write a summary of Las Vegas."


# In[ ]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 100,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[ ]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[ ]:


print(json.dumps(response_body, indent=4))


# In[ ]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body" : json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 500,
                "temperature": 0.7,
                "topP": 0.9
            }
        }
    )
}


# In[ ]:


response = bedrock_runtime.invoke_model(**kwargs)
response_body = json.loads(response.get('body').read())

generation = response_body['results'][0]['outputText']
print(generation)


# In[ ]:


print(json.dumps(response_body, indent=4))


# ### Working with other type of data

# In[ ]:


from IPython.display import Audio


# In[ ]:


audio = Audio(filename="dialog.mp3")
display(audio)


# In[ ]:


with open('transcript.txt', "r") as file:
    dialogue_text = file.read()


# In[ ]:


print(dialogue_text)


# In[ ]:


prompt = f"""The text between the <transcript> XML tags is a transcript of a conversation. 
Write a short summary of the conversation.

<transcript>
{dialogue_text}
</transcript>

Here is a summary of the conversation in the transcript:"""


# In[ ]:


print(prompt)


# In[ ]:


kwargs = {
    "modelId": "amazon.titan-text-express-v1",
    "contentType": "application/json",
    "accept": "*/*",
    "body": json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 512,
                "temperature": 0,
                "topP": 0.9
            }
        }
    )
}


# In[ ]:


response = bedrock_runtime.invoke_model(**kwargs)


# In[ ]:


response_body = json.loads(response.get('body').read())
generation = response_body['results'][0]['outputText']


# In[ ]:


print(generation)

