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

        # Utilize my GPU (RTX 3060)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto")

        # Suppress warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


    def generate_response(self, prompt: str) -> str:
        # Tokenize inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate model output ids
        output = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=10)
        output_token_ids = output.sequences[0]

        # Decode output
        decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # Remove prompt from generated text (could be done by masking or indexing based off prompt string)
        generated_text = decoded_output[len(prompt):].strip()

        # Obtain action by manipulating the generated text (mistral outputs action in the form of "Action: > action")
        generated_text = generated_text.split('\n')[0][2:]

        # Print prompt and action for debugging
        print("[PROMPT] " + prompt.replace("Action:", ""))
        print("[ACTION] " + generated_text)

        return generated_text

    
    def generate_constrained_response(self, prompt: str, valid_actions: List[str]) -> str:

        prompt = prompt.replace("Action:", "") # Remove "Action:" from prompt

        for action in valid_actions:
            # Task-grounding
            relevant_prompt = prompt + f"Q: Given the action {action}, what is the probability from 0 to 1 that this action is relevant in the long-term to our goal? A: The probability is"

            inputs = self.tokenizer(relevant_prompt, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=10)
            output_token_ids = output.sequences[0]

            decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

            print(decoded_output)



        

        

        

        # valid_action_scores = []
        # modified_prompt = prompt.replace("Action:", "")

        # for action in valid_actions:
        #     print("Looking at action: ", action)
        #     # "Task-grounding", according to authors
        #     relevance_prompt = modified_prompt + f"\nGiven the action {action}, what is the probability from 0 to 1 that this action is relevant in the long-term to our goal?"

        #     # "World-grounding", according to authors
        #     affordance_prompt = modified_prompt + f"\nGiven the action {action}, what is the probability that this action is feasible in the current environment?"

        #     # Tokenize inputs
        #     relevance_inputs = self.tokenizer(relevance_prompt, return_tensors="pt").to(self.device)
        #     affordance_inputs = self.tokenizer(affordance_prompt, return_tensors="pt").to(self.device)

        #     # Generate model output ids
        #     relevance_output = self.model.generate(**relevance_inputs, return_dict_in_generate=True, max_new_tokens=10)
        #     affordance_output = self.model.generate(**affordance_inputs, return_dict_in_generate=True, max_new_tokens=10)

        #     # Decode output
        #     relevance_decoded_output = self.tokenizer.decode(relevance_output.sequences[0], skip_special_tokens=True)
        #     affordance_decoded_output = self.tokenizer.decode(affordance_output.sequences[0], skip_special_tokens=True)



