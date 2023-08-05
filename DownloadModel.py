from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

load_dotenv()

tokenizer = AutoTokenizer.from_pretrained(
    os.getenv("MODEL_ID")
    use_fast=True
)

config = AutoModelForCausalLM.from_pretrained(os.getenv("MODEL_ID"))