from dotenv import load_dotenv
from ctransformers import AutoTokenizer, AutoModelForCausalLM
import os

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    use_fast=True
)

config = AutoModelForCausalLM.from_pretrained(MODEL_ID)