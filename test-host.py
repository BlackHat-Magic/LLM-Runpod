from dotenv import load_dotenv
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
import runpod, os, argparse

load_dotenv()

#runpod.api_key = os.getenv("RUNPOD_API_KEY")
#pygmalion_endpoint = runpod.Endpoint(os.getenv("PYGMALION_ENDPOINT"))

tokenizer = AutoTokenizer.from_pretrained(
    os.getenv("MODEL_ID"),
    use_fast=True
)

config = AutoConfig.from_pretrained(
    os.getenv("MODEL_ID"), 
    trust_remote_code=True
)
config.max_position_embeddings = 8192

model = AutoModelForCausalLM.from_pretrained(
    os.getenv("MODEL_ID"),
    trust_remote_code=True,
    device_map="auto"
)

job_input = {
    "messages": [
        {
            "role": "assistant",
            "content": "Hello!"
        },
        {
            "role": "user",
            "content": "Hi there!"
        }
    ]
}

input_ids = None
prompt = ""
for message in job_input["messages"]:
    if(input_ids != None and len(input_ids) > 7680):
        break
    if(message["role"] == "assistant"):
        prompt += f"ASSISTANT: {message['content']}\n"
    elif(message["role"] == "user"):
        prompt += f"USER: {message['content']}\n"
    else:
        prompt += f"SYSTEM: {message['content']}\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

output = model.generate(
    inputs=input_ids, 
    temperature=job_input.get("temperature", 0.7),
    max_new_tokens=job_input.get("max_response_length", (8192 - len(input_ids))),
)

print(tokenizer.decode(output[0]))
