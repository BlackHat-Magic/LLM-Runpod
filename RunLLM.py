from dotenv import load_dotenv
from ctransformers import AutoModelForCausalLM
import runpod, os

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")

model = AutoModelForCausalLM.from_pretrained(
    os.getenv("MODEL_ID"),
    model_type="llama"
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
            return_tensores="pt"
        ).input_ids.cuda()
    while len(input_ids > 0.75 * config.max_position_embeddings):
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
            return_tensores="pt"
        ).input_ids.cuda()

    output = mode.generate(
        inputs=input_ids, 
        temperature=job_input.get("temperature", float(os.getenv("DEFAULT_TEMPERATURE"))),
        max_new_tokens=job_input.get(
            "max_response_length", 
            (config.max_position_embeddings - len(input_ids))
        ),
    )

    return(tokenizer.decode(output[0]))

runpod.serverless.start({"handler": generate_text})
