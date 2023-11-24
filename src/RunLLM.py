from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
import runpod, os

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
CACHE_DIR = os.getenv("CACHE_DIR")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
MAX_TOKEN_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH"))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

def generate_text(job):
    job_input = job["input"]
    prompt = ""
    input_ids = None
    messages = job_input["messages"]
    for message in messages:
        if(message["role"] == "assistant"):
            prompt += f"ASSISTANT: {message['content']}\n"
        elif(message["role"] == "user"):
            prompt += f"USER: {message['content']}\n"
        else:
            prompt += f"{message['content']}\n"
        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.cuda()
    while len(input_ids > 0.75 * MAX_TOKEN_LENGTH):
        messages.pop(1)
        for message in messages:
            if(message["role"] == "assistant"):
                prompt += f"ASSISTANT: {message['content']}\n"
            elif(message["role"] == "user"):
                prompt += f"USER: {message['content']}\n"
            else:
                prompt += f"SYSTEM: {message['content']}\n"
        input_ids = tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.cuda()

    output = mode.generate(
        inputs=input_ids, 
        temperature=job_input.get("temperature", DEFAULT_TEMPERATURE),
        max_new_tokens=job_input.get(
            "max_response_length", 
            (config.max_position_embeddings - len(input_ids))
        ),
    )

    return(tokenizer.decode(output[0]))

runpod.serverless.start({"handler": generate_text})
