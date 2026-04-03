
#Imports Python’s os module.
#i use it to read environment variables like OPENAI_API_KEY securely (instead of hardcoding the key).
import os
#Imports json for converting Python dictionaries into JSON text.
import json
#Imports datetime so i can timestamp every run for   experiment tracking.
from datetime import datetime
#This lets Python read variables from a local .env file and load them into environment variables.
from dotenv import load_dotenv
from openai import OpenAI
#Reads your .env file
load_dotenv()

#create API client used to call the LLM.
#This is the “LLM interface” used for  baseline generation.
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL_NAME="gpt-4o-mini"
#Controls randomness.0 makes the output as deterministic as possible.Thesis meaning:For a control experiment, fixing temperature helps ensure outputs are consistent and comparable.
TEMPERATURE=0.7

#Defines a function that takes a string prompt and returns the model’s answer by Sending the prompt to the model using OpenAI’s Responses API .The returned object response contains the model output in a structured format 

def generate_answer(prompt:str):
    response=client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        temperature=TEMPERATURE
    )
    #Extract output text ,so first check it is a message (i.e., conversational text output),A “message” can contain multiple content blocks.Example: text, images, etc. so again we loop to get output_text only
    output_text=""
    for item in response.output:
        if item.type=="message":
            for content in item.content:
                if content.type=="output_text":
                    output_text+=content.text
    return output_text               


#Defines a function that logs one experiment record.(control experiment record)

def log_result(prompt: str, output: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),#current time in UTC in ISO format (standard format for experiments)
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "prompt": prompt,
        "output": output
    }
    #Opens logs/results.jsonl in append mode "a" and Converts the dictionary into JSON text with json.dumps(log_entry).
    with open("logs/results.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

