from env import Env, State, Robot


def main():
    # initialize a new environment 
    env = Env()
    # create a start state with an empty inventory
    start_state = State(env=env, 
                        location="living room", 
                        inventory=[])

    # get a robot object for interacting with the env 
    robot = Robot(env=env,
                  system_prompt="You are an intelligent robot. Your goal is to drop a knife in the living room. You can navigate the environment, pick up items, and drop them.",
                  current_state=start_state)

    # manual interaction with the env
    print(robot.observe())  
    while True:
        action = input("Enter an action: ")
        if action == "exit":
            break
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
    main()