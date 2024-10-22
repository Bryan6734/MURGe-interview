import os 
from typing import List 

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")

class Agent:
    def __init__(self,
                 model_name) -> None:

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # Suppress warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


    def generate_response(self, prompt: str) -> str:
        # Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate model output ids
        output = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=5)
        output_token_ids = output.sequences[0]

        # Decode output
        decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # Remove prompt from generated text (could be done by masking or indexing based off prompt string)
        generated_text = decoded_output[len(prompt):].strip()

        return generated_text

    
    def generate_constrained_response(self, prompt: str, valid_actions: List[str]) -> str:
        # TODO: this function takes a prompt and a list of valid actions
        # it should return the best action according to the model, as in the SayCan paper 

        # Turn valid actions into a prompt
        valid_actions_str = " ".join(valid_actions)
        prompt = f"{prompt} Valid actions: {valid_actions_str}. Action:"
        

        # Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate model output ids
        output = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=5)
        output_token_ids = output.sequences[0]

        # Decode output
        decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # Remove prompt from generated text (could be done by masking or indexing based off prompt string)
        generated_text = decoded_output[len(prompt):].strip()

        return generated_text


# agent = Agent("gpt2-medium")

# instructions = "You are an intelligent robot. Your goal is to drop a knife in the living room. Knife is in the kitchen. You can navigate the environment, pick up items, and drop them. You are in the living room. You see: couch, television, book. You have the following items in your inventory: . Valid actions: pick up couch, pick up television, pick up book, go to bathroom, go to kitchen, go to bedroom. Goal: Your goal is to drop a knife in the living room. Put the knife in your inventory, then navigate to the living room and drop the knife. Action:"

# action = agent.generate_response(instructions)
# # print("[Action] " + )
# # action[0]
# print("[PROMPT]", instructions)
# print("------")
# print("[ACTION]", action)
