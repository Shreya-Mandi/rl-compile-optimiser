from env import GraphEnv
import numpy as np

def reward_function(state, action):
    # Custom reward logic based on state and action TODO
    return -1 if state != action else 0

def train(env, epochs, learning_rate=0.1, discount_factor=0.95):
    # Initialize Q-table with zero
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    for epoch in range(epochs):
        state = env.reset()
        done = False

        while not done:
            # Choose action from Q table (greedy)
            if np.random.rand() < 0.1:  # Exploration with epsilon=0.1
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            # Take the action and observe the new state and reward
            next_state, reward, done, _ = env.step(action)

            # Update Q-table using the Bellman equation
            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += learning_rate * td_error

            # Update state
            state = next_state

        # Optionally, print the Q-table every epoch
        print(f"Epoch {epoch}: Q-table \n{Q}")

    return Q

def test(env, input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()
    # Example of processing input and managing state transitions
    # Log results to an output file
    with open(output_file, 'w') as file:
        file.write("Test results")

# Example graph: node dictionary with rewards
graph = {0: {'reward': -1}, 1: {'reward': -1}, 2: {'reward': 10}}

env = GraphEnv(graph, 0)
train(env, epochs=10)
test(env, 'input.txt', 'output.txt')