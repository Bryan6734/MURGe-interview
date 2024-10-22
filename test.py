import os 
from typing import List 

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

# CACHE_DIR = os.environ.get("HF_HOME", None)

class Agent:
    def __init__(self,
                 model_name) -> None:

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_response(self, prompt: str) -> str:
        # TODO: this function takes a prompt and return a response
        # use model.generate() to generate a response without constraints 
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=50, return_dict_in_generate=True, output_scores=True)

        generated_token_ids = outputs.sequences[0]
        new_output = self.tokenizer.decode(generated_token_ids)

        return new_output[0]

    def generate_constrained_response(self, prompt: str, valid_actions: List[str]) -> str:
        # TODO: this function takes a prompt and a list of valid actions
        # it should return the best action according to the model, as in the SayCan paper 
        pass 

agent = Agent("gpt2")

response = agent.generate_response("2 + 2 is equal to")
print(response[0])