import os 
from typing import List 

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE_DIR = os.environ.get("HF_HOME", None)

class Agent:
    def __init__(self,
                 model_name) -> None:
        # TODO: load the model and tokenizer
        pass 

    def generate_response(self, prompt: str) -> str:
        # TODO: this function takes a prompt and return a response
        # use model.generate() to generate a response without constraints 
        pass 

    def generate_constrained_response(self, prompt: str, valid_actions: List[str]) -> str:
        # TODO: this function takes a prompt and a list of valid actions
        # it should return the best action according to the model, as in the SayCan paper 
        pass 