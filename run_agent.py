import argparse
from env import Env, State, Robot
from agent import Agent

import numpy as np
import torch
import random

np.random.seed(12)
torch.manual_seed(12)
random.seed(12)

def main(args):
    # initialize a new environment 
    env = Env()
    # create a start state with an empty inventory 
    start_state = State(env=env, 
                        location="living room", 
                        inventory=[])

    # the rationale is needed to guide the model towards the right solution  
    rationale_str = """Goal: Your goal is to drop a knife in the living room. 
Put the knife in your inventory, then navigate to the living room and drop the knife."""
    # instantiate robot object 
    robot = Robot(env=env,
                  system_prompt="You are an intelligent robot. Your goal is to drop a knife in the living room. Knife is in the kitchen. You can navigate the environment, pick up items, and drop them.",
                  current_state=start_state,
                  rationale_str=rationale_str)

    print(robot.observe())  
    # initialize the agent
    # the first time this runs it will take a while to download the model 
    # after that it should be fairly quick (5-10 seconds)
    # you should be able to run this on a laptop with a latency of about 5 seconds per action 
    # if you have a GPU, it will be faster
    # if you want to develop more quickly, you can use GPT2 for debugging
    # but its performance will likely be poor 
    agent = Agent(model_name="mistralai/Mistral-7B-v0.1")
    # agent = Agent(model_name="gpt2") 


    max_steps = 10
    for _ in range(max_steps):
        if args.constrained:
            action = agent.generate_constrained_response(robot.observe(), robot.current_state.get_valid_actions())
        else:
            action = agent.generate_response(robot.observe())

        print(f"ACTION: {action}")
        print()
        try:
            robot.execute_action(action)
        except ValueError as e:
            print(e)
        print(robot.observe())

        # check success 
        if 'knife' in env.env_lut['living room']: 
            print("Congratulations! You have successfully placed the knife in the living room!")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--constrained", action="store_true", help="Use the constrained generation method")
    args = parser.parse_args()
    main(args)