# MURGE-Lab summer position interview

Please read all of the instructions below before starting.

## Interview structure
There will be two parts to the interview. The first component, which is contained in this folder, is a take-home assessment that is designed to reflect the kind of tasks that a research assistant would be asked to do. You will be assessed based on the quality of your solution. If you pass the second assessment, you will be invited for an in-person interview. In that interview, you'll be asked about your past experience and your skills, and also be asked to perform a similar coding task to the one contained here. 

Both programming challenges involve using LLMs to obtain model scores. Both will rely on Huggingface's Transformers library. This exercise is not only a chance for you to demonstrate your coding ability, but also an opportunity to get up-to-speed on the library, which you will need in any NLP role, especially in MURGE-Lab. 

## The task
This task will involve controlling a simple simulated robot in an interactive environment. To make your life easier, we have provided the environment code in this directory. 

## Task 1: Reading
Some of the key ideas for this task are inspired by recent research on controlling robots with LLMs. Please read [this paper](https://say-can.github.io/assets/palm_saycan.pdf) before proceeding. After reading the paper, you should be able to summarize the key problem the paper is trying to solve, how they approach the problem, and what their results are. You may need to go into the paper's references to understand some of the background. You can expect to be asked a few questions about this paper in the second interview. 

The task given to you here focuses on LLM scoring, so make sure you understand the language model component of SayCan (as opposed to the value function, which will not be included in this task.)

## Environment 
In `run_agent.py` there is code to run an LLM-based agent in a very simple environment. There is only one goal in the game: to drop a knife in the living room. Actions are strings of text like "go to living room" and "pick up knife". You can play the game manually in `main.py`. 

There are some rules and constraints to the game.
1. Any `pick up` or `drop` action must be followed by a `go to` action.
2. Any `go to` action must be followed by a `pick up` action.
3. `pick up x` results in `x` being added to your inventory and being removed from the room you are in.
4. `drop x` results in `x` being added to the room and removed from your inventory. 
5. you can only pick up things in the room you're in, and you can only drop items you have picked up. 

## Task 2: Generation 
Your task will be to implement the `Agent` class in `agent.py`. 
You'll submit several files with outputs and answers to some short questions. 
Please make sure to include your `agent.py` file when you submit. 

The class should take a model name and should load the model using Huggingface's Transformers library. 
The class has two methods. The first is `generate_response()` which generates a response given a prompt. 
The prompt is provided to you by the environment, and you can inspect it in `run_agent.py`. 

`generate_response()` takes the prompt and generates a response from the model. This generation is done without any constraints. 
By default, generation should be done with the `mistralai/Mistral-7B-v0.1` LLM. All the initalization code is set up already in `run_agent.py` so you should only have to fill in `agent.py`.

After implementing `generate_response()`, test it by running `python run_agent.py`. Copy your output and save it as `unconstrained_output.txt`. 
The actions your model outputs should be strings, though not necessarily valid actions. 

Q1: What problems do you notice with this approach? If it is not working, why do you think it is not working? Write your response (3-4 sentences max) and save it in a file called `q1_response.txt`.A

## Task 3: Constrained generation
The environment has a function called `get_valid_actions()` that can be found in `env.py`. 
This returns the valid actions available to the agent. 
Your second task is to implement a scoring-based approach, much like the approach taken in the SayCan paper. 
Specifically, you will implement the `generate_constrained_response()` function in `agent.py`. This function takes a prompt and a set of valid actions.
It returns one of the actions from the set of input actions. 
You can run it by calling `python run_agent.py --constrained`.

Paste the output of your agent into `constrained_output.txt`.
Your constrained agent should be able to solve the task provided in `run_agent.py`. 

Q2: Why did constrained generation work? Save your response in `q2_response.txt`
Q3: Are there any major limitations to the constrained generation approach you have implemented? Save your response in `q3_response.txt` 

## Submitting
Please zip all of these files together and submit them via email to esteng@cs.unc.edu
You should not have to modify `run_agent.py` or `env.py`. 