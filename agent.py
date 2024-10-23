import os 
from typing import List 

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings


class Agent:
    def __init__(self,
                 model_name) -> None:

        # Utilize my GPU (RTX 3060)
        # Had to install other librarie (accelerate, bitsandbytes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True, device_map="auto") # quantized model

        # Suppress warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")

    """
    Generate an unconstrained response given a prompt

    """
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

    """
    Generate a constrained response given a prompt and a list of valid actions
    
    1. Score each action based on the chance that the action is relevant to the goal (scored between 0 and 10, because model struggled with generating probabilities)
    2. Return the action with the highest score 
    
    """
    def generate_constrained_response(self, prompt: str, valid_actions: List[str]) -> str:

        prompt = prompt.replace("Action:", "")
        relevance_scores = []

        for action in valid_actions:
            
            relevance_score = self.score_action(prompt, action)

            print("[ACTION] " + action)
            print("[RELEVANCE] " + str(relevance_score))

            relevance_scores.append(relevance_score)
        
        # Return action with highest relevance score
        max_score = max(relevance_scores)
        max_score_index = relevance_scores.index(max_score)
        print("-------------")
        print("[MAX SCORE] " + str(max_score))
        print("[MAX SCORE ACTION] " + valid_actions[max_score_index])
        return valid_actions[max_score_index]

    def score_action(self, prompt: str, action: str) -> float:
        # Construct prompt for relevance. Followed HuggingFace's prompting best practices on the website. 
        relevant_prompt = "Context: " + prompt + f"Q: Given the action {action}, how relevant is this action to your goal on a scale of 0 to 100? A: Given our goal, I would give it a score of "

        # Generate model response
        inputs = self.tokenizer(relevant_prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, return_dict_in_generate=True, max_new_tokens=3)
        output_token_ids = output.sequences[0]

        # Obtain action relevance score
        decoded_output = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)

        # Extract relevance score from model output
        relevance_score = decoded_output.split("A: ")[1].strip()
        
        # Convert to float
        relevance_score = ''.join(filter(str.isdigit, relevance_score))
        relevance_score = float(relevance_score)

        return relevance_score

