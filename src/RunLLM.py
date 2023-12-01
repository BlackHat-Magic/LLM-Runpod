from dotenv import load_dotenv
# from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import runpod, os

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
CACHE_DIR = os.getenv("CACHE_DIR")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE"))
MAX_TOKEN_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH"))

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     cache_dir=CACHE_DIR,
#     device_map="auto",
#     trust_remote_code=True
# )
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

model = LLM(
    model=MODEL_ID,
    download_dir=CACHE_DIR,
    trust_remote_code=True
)

def generate_text(job):
    job_input = job["input"]
    prompt = ""
    input_ids = None
    messages = job_input["messages"]
    for message in messages:
        prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    while len(input_ids) > 0.75 * MAX_TOKEN_LENGTH:
        messages.pop(1)
        prompt = ""
        for message in messages:
            prompt += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    latest_message_tokens = tokenizer(messages[-1]["content"], return_tensors="pt")
    max_new_tokens = MAX_TOKEN_LENGTH - len(input_ids)
    if(job_input.get("max_response_length", False)):
        max_new_tokens = min(job_input["max_response_length"], max_new_tokens)

    temperature = DEFAULT_TEMPERATURE
    if(job_input.get("temperature", False)):
        temperature = job_input.get("temperature")
    
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)

    outputs = model.generate(
        [prompt], 
        sampling_params
    )

    return(outputs[0].outputs[0].text)
    # return(prompt)

runpod.serverless.start({"handler": generate_text})
