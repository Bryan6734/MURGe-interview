from typing import List 

class Env:
    def __init__(self) -> None:
        self.env_lut = {"kitchen": ["knife", "fork", "spoon"],
           "living room": ["couch", "television", "book"],
           "bedroom": ["bed", "pillow", "blanket"],
           "bathroom": ["toothbrush", "sink", "shower"]}

class State:
    def __init__(self,
                 env: Env,
                 location: str,
                 inventory: List[str]) -> None:
        self.env = env
        self.location = location
        self.inventory = inventory

        self.history = ["no past actions"]

    def get_valid_actions(self) -> List[str]:
        valid_actions = []
        last_action = self.history[-1]
        # Rules: no two go to or pickup/drop actions in a row 
        add_go_to, add_pickup_drop = False, False
        if last_action.startswith("go to"):
            add_pickup_drop = True
        elif last_action.startswith("pick up") or last_action.startswith("drop"):
            add_go_to = True
        elif last_action == "no past actions":
            # for first action add both 
            add_pickup_drop = True
            add_go_to = True
        else:
            raise ValueError(f"Invalid action in history: {last_action}")

        if add_pickup_drop:
            # next action has to be pick up or drop 
            for item in self.env.env_lut[self.location]:
                valid_actions.append(f"pick up {item}")
            for item in self.inventory:
                valid_actions.append(f"drop {item}")
        if add_go_to:
            # alternate pick up and go to actions 
            for location in self.env.env_lut.keys() - {self.location}:
                valid_actions.append(f"go to {location}")

        return valid_actions
    
    def get_observation(self) -> str:
        return f"""You are in the {self.location}. 
You see: {', '.join(self.env.env_lut[self.location])}.
You have the following items in your inventory: {', '.join(self.inventory)}."""

class Robot:
    def __init__(self,
                 env: Env,
                 system_prompt: str,
                 current_state: State,
                 rationale_str: str = None) -> None:
        self.env = env
        self.system_prompt = system_prompt
        self.current_state = current_state
        self.rationale_str = rationale_str

    def observe(self) -> str:
        prompt = f"""{self.system_prompt}
{self.current_state.get_observation()}
Valid actions: {', '.join(self.current_state.get_valid_actions())}
{self.rationale_str}
Action:"""
        return prompt

    def execute_action(self, action: str) -> None:
        if action not in self.current_state.get_valid_actions():
            raise ValueError(f"Invalid action: {action}")

        # execute action
        if action.startswith("go to"):
            target = action.split("go to ")[1]
            # if the target is invalid, raise error
            if target not in self.env.env_lut.keys():
                raise ValueError(f"Invalid location: {target}")
            # update the state 
            self.current_state.location = target

        elif action.startswith("pick up"): 
            target = action.split("pick up ")[1]
            # if the target is invalid, raise error
            if target not in self.env.env_lut[self.current_state.location]:
                raise ValueError(f"Invalid item: {target}")
            # add the item to the inventory
            self.current_state.inventory.append(target)
            # remove the item from the environment 
            self.env.env_lut[self.current_state.location].remove(target)

        elif action.startswith("drop"):
            target = action.split("drop ")[1]
            # if the target is invalid, raise error
            if target not in self.current_state.inventory:
                raise ValueError(f"Invalid item: {target}")
            # remove the item from the inventory
            self.current_state.inventory.remove(target)
            # add the item to the environment
            self.env.env_lut[self.current_state.location].append(target)

        else:
            raise ValueError(f"Invalid action: {action}")

        # update history 
        self.current_state.history.append(action)
