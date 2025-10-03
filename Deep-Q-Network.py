# Deep-Q-Network 
    # DQN - Deep Q Network is the combination of Reinforcement learning and Deep learning. 
    # It’s designed to solve complex decision-making problems where the state or action space is too large 
        # to be handled by traditional Q-learning with lookup tables.
    # DQN Architecture 
        # Input Layer: Takes the current state s (e.g., pixel frame or numerical vector).
        #  Hidden Layers: Deep neural layers (e.g., ReLU-activated dense or convolutional).
        # Output Layer: Outputs a vector of Q-values [Q(s, a₁), Q(s, a₂), ..., Q(s, aₙ)], one for each possible action.
    # DQN stores (state, action, reward, next_state, done) in a buffer.
    # During training, samples mini-batches randomly to break temporal correlation.
    # Maintains two networks:
        # Q-network (learned)
        # Target Q-network (slowly updated)
    # Reduces oscillations and divergence by using a stable target in the Bellman update.
    # Reward clipping: Helps keep training stable by avoiding large reward explosions.
    # Input normalization: Especially for image-based state inputs.
    # DQN Training cycle 
        # Initialize Q-network and Target Q-network with random weights.
        # For each episode:
            # Get the state s(t) 
            # For each step:
                # Choose the action a under the policy π. 
                # Take the action to get the rewards r(t) and next state s(t+1)
                # Store (s(t), a(t), r(t), s(t+1), done) in replay memory.
                # Sample mini-batch from memory.
                # Compute the target values using the values from the mini-batch
                    # y(t) = r(t) + γ [max(Qtarget(s(t-1),a(t-1)))]
                        # Where
                            # y(t) - Target value
                            # r(t) - expected reward
                            #  γ - discount factor
                # Update the Q network 
                    # Loss(L) = Q(s(t), a(t)) - y(t)
                        # Where,    
                            # L - loss function, 
                            # Q(s(t), a(t)) - Calculated Q funtion on time t
                            # y(t) - target value
                # For each step, update the target network 
    # The loss function is calculated by the mean squared error of the current and targeted values. 
    # Hyper-parameters 
        # | Hyperparameter       | Description                             |
        # | -------------------- | --------------------------------------- |
        # | `γ (gamma)`          | Discount factor (e.g., 0.99)            |
        # | `ε`                  | Exploration rate for ε-greedy (decayed) |
        # | `Replay buffer size` | How many experiences to store           |
        # | `Batch size`         | Size of training batch (e.g., 32, 64)   |
        # | `Learning rate`      | For optimizer (e.g., 0.00025)           |
        # | `Target update`      | How often to update target network      |
    # Pseudocode
        # Initialize q_network and target_q_network (same architecture)
        # Initialize replay_buffer, optimizer, loss_fn
        # Set γ (discount), ε (exploration), C (target update frequency)

        # for episode in range(num_episodes):
        #     s = env.reset()
        #     for t in range(max_steps):
        #         With probability ε, select random action a
        #         else a = argmax_a q_network(s)
                
        #         Execute a → observe (r, s', done)
        #         Store (s, a, r, s', done) in replay_buffer
                
        #         s = s'

        #         If replay_buffer has enough samples:
        #             Sample minibatch from replay_buffer

        #             For each sample:
        #                 Compute target Q-value:
        #                     if done: y = r
        #                     else: y = r + γ * max(target_q_network(s'))

        #             Compute predicted Q(s, a) from q_network
        #             Compute loss = MSE(y, Q(s, a))

        #             Update q_network weights using gradients

        #         Every C steps:
        #             target_q_network ← q_network (soft or hard copy)

        #         If done → break

# Enviroment 
import numpy as np
import tensorflow as tf 
from collections import deque
import random

class GridWorld:
    def __init__(self, rows=4, cols=4, goal=(3, 3)):
        self.rows = rows
        self.cols = cols
        self.goal = goal
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]  # Start at top-left
        return self._get_state()
    
    def _get_state(self):
        # Convert agent position to a single state number
        # [row, col] -> state number
        # col -> particualar column in the grid, row -> particular row in the grid
        # state number = row * no_cols + col
        return (self.agent_pos[0] * self.cols) + self.agent_pos[1]
    
    def step(self, action):
        # Actions: 0 = Up, 1 = Right, 2 = Down, 3 = Left
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.cols - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.rows - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1

        state = self._get_state()
        done = (tuple(self.agent_pos) == self.goal)
        reward = 10 if done else -1
        return state, reward, done

# Hyperparameters 

# Number of states 
state_dim = 16

# No of actions to take 
action_dim = 4

# Learning rate for the optimizer
learning_rate = 0.0005

# Discount factor for future rewards
gamma = 0.99

# Initial exploration rate (ε)
epsilon = 1.0

# Final (minimum) exploration rate
epsilon_min = 0.01

# Decay rate for exploration (linear or exponential)
epsilon_decay = 0.995  # Multiplied after each episode or step

# Number of experiences to sample per training step
batch_size = 64

# Total size of the experience replay buffer
replay_buffer_capacity = 100_000

# Frequency (in steps) to update the target network
target_update_freq = 100  # Hard update every N steps

# Number of steps before learning starts (to fill buffer)
train_start = 1000

# Max steps per episode (optional control)
max_steps_per_episode = 500

# Total number of training episodes
num_episodes = 1000

# Seed for reproducibility (if needed)
random_seed = 42

# reply buffer to store the current and next state values (state, action, rewards, next_state, goal_reached_status)
replay_buffer = deque(maxlen=replay_buffer_capacity)

# Sample batch
def sample_batch(buffer, batch_size):
    return random.sample(buffer, batch_size)

# Method to create a network 
def createNetwork(inputDimension : int, outputDimension : int) -> tf.keras.Model:
    try:
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(inputDimension,)),
            tf.keras.layers.Dense(inputDimension // 2, activation='relu'),
            tf.keras.layers.Dense(inputDimension // 4, activation='relu'),
            tf.keras.layers.Dense(outputDimension)  # Q-values for each action
        ])
    except Exception as e:
        print(f"An error occoured {e}")
        return None

# update the Q_Network by calculate the target value and loss function 
    # Compute the target values using the values from the mini-batch
        # y(t) = r(t) + γ [max(Qtarget(s(t-1),a(t-1)))]
            # Where
                # y(t) - Target value
                # r(t) - expected reward
                #  γ - discount factor
    # Update the Q network 
         # Loss(L) = Q(s(t), a(t)) - y(t)
                # Where,    
                    # L - loss function, 
                    # Q(s(t), a(t)) - Calculated Q funtion on time t
                    # y(t) - target value
# calculate the target value 
def calculate_targetValue(reward : int, q_values : np.ndarray):
    try:
        # return the calculated target value
        return reward + gamma * np.max(q_values)
    except Exception as e:
        print(f"Error occoured in target value : {e}")   
        return None          

# loss function
def calculate_Q_Loss(reward : int, q_values_next : np.ndarray, state : any, action: any, Q_Network : tf.keras.Model):
    try:
        # forward propagation to get the action from the current policy 
        Q_values = Q_Network(tf.expand_dims(
                tf.convert_to_tensor(
                    tf.one_hot(state, depth=16, dtype=tf.float32), 
                    dtype=tf.float32
                    ), 
                axis=0
                )
            )

        # Gather Q(state, action) for the action taken 
        Q_Selected = tf.reduce_sum(Q_values * tf.one_hot(action, action_dim), axis=1)

        # Invoke the calculate_targetValue to get the target value 
        target_value = calculate_targetValue(reward=reward, q_values=q_values_next)

        # check if the target value is none 
        if target_value is None:
            # raise the expcetion 
            raise Exception("target value is None")
        
        # return the mean squared error between the current q value and target value 
        return (target_value - Q_Selected) ** 2

    except Exception as e:
        print(f"Error occoured in loss calculation : {e}")    
        return None    

# udpate Q network
def update_Q_Network(reward : int, 
                     q_values_next : np.ndarray, 
                     state : any, action : any, 
                     Q_Network: tf.keras.Model, 
                     optimizer : any):
    try:
        # Calculate the gradients inside the gradient tape to have the record operations for the forward and backward pass 
        with tf.GradientTape() as gradientTape:
            # Calculate the mse 
            Q_Loss = calculate_Q_Loss(
                reward=reward, 
                q_values_next=q_values_next, 
                state=state, action=action,
                Q_Network=Q_Network
                )

        # Apply gradients to update weights
        # calculate the gradient using the mse and network trainable variables
        grads = gradientTape.gradient(Q_Loss, Q_Network.trainable_variables)

        # train the weights by invoking the apply gradient in optimizer 
        optimizer.apply_gradients(zip(grads, Q_Network.trainable_variables))
    except Exception as e:
        print(f"Error occoured in Update Q Network : {e}")


# Initialize the enviroment as env 
env = GridWorld()

# step function
    # Choose the action a under the policy π. 
    # Take the action to get the rewards r(t) and next state s(t+1)
    # Store (s(t), a(t), r(t), s(t+1), done) in replay memory.
    # Sample mini-batch from memory.
    # Compute the target values using the values from the mini-batch
        # y(t) = r(t) + γ [max(Qtarget(s(t-1),a(t-1)))]
            # Where
                # y(t) - Target value
                # r(t) - expected reward
                #  γ - discount factor
    # Update the Q network 
         # Loss(L) = Q(s(t), a(t)) - y(t)
                # Where,    
                    # L - loss function, 
                    # Q(s(t), a(t)) - Calculated Q funtion on time t
                    # y(t) - target value
    # For each step, update the target network  
def step_function(max_steps_per_episode : int = state_dim, 
                  Q_Network : tf.keras.Model = None, 
                  Target_Q_Network : tf.keras.Model = None, 
                  optimizers : any = None,
                  state : any = None, 
                  episode_reward : int = 0, 
                  global_step : int = 1
                  ):
    try:

        for _ in range(max_steps_per_episode):
             
            # Initialize the action 
            action = 0

            # Create a 16-element one-hot vector
            oneHotTensor = tf.one_hot(state, depth=16, dtype=tf.float32)

            # print(f"one hot tensor : {oneHotTensor}")

            # print(tf.expand_dims(
            #         tf.convert_to_tensor(oneHotTensor, dtype=tf.float32), 
            #         axis=0
            #     ))

            # get the q_value from the Q_network and pass the state input 
            q_value = Q_Network(
            tf.expand_dims(
                tf.convert_to_tensor(oneHotTensor, dtype=tf.float32), 
                axis=0
                )
            ) # state input

            # selecting the action in the given policy 
            if np.random.rand() < epsilon:
                # select the random action to explore
                action = np.random.choice(action_dim)
            else:
                # Select the action by taking the maximum argument to exploit 
                action = tf.argmax(q_value[0]).numpy()

            # print the action 
            # print(f"Action : {action}")

            # Take the action in the enviroment 
            next_state, reward, done = env.step(action)
            
            # Store the exprience (state, action, reward, next_state, done, Qvalue) in replay buffer 
            replay_buffer.append((state, action, reward, next_state, done, q_value))

            # Check if the reply_buffer is greater than batch size 
            if len(replay_buffer) >= batch_size:
                # print("Gonna update Q Network !!!!")

                # train the network with the data in reply buffer 
                update_Q_Network(
                    reward=episode_reward,
                    q_values_next=Q_Network(
                        tf.expand_dims(
                            tf.convert_to_tensor(
                                tf.one_hot(next_state, depth=16, dtype=tf.float32), 
                                dtype=tf.float32
                                ), 
                            axis=0
                        )
                    ),
                    state=state,
                    action=action,
                    Q_Network=Q_Network,
                    optimizer=optimizers
                )

            # update the target q network for the based on the target frequence 
            if global_step % target_update_freq == 0:
                # update the weights 
                 Target_Q_Network.set_weights(Q_Network.get_weights())

            # update the state 
            state = next_state
            episode_reward += reward
            global_step += 1

            # break if the goal is reached by check the done 
            if done:
                break

    except Exception as e:
        print(f"Exception in step function : {e}")   
        return 

# DQN architecture
    # Initialize q_network and target_q_network (same architecture)
        # Initialize replay_buffer, optimizer, loss_fn
        # Set γ (discount), ε (exploration), C (target update frequency)
        
        # for episode in range(num_episodes):
            # env.reset()
            # initialize the Step function
def DQN():
    try:
        # Initialize the networks
        # create Q-network and target q network 
        Q_Network = createNetwork(inputDimension=state_dim, outputDimension=action_dim)
        Target_Q_Network = createNetwork(inputDimension=state_dim, outputDimension=action_dim)

        # Initialize the same weight for the target network 
        Target_Q_Network.set_weights(Q_Network.get_weights())

        # Initialize the optimizer and loss functions 
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # run the episodes 
        for _ in range(num_episodes):

            # print("Start of the episode !!!")

            # Initalize the necessary variables  
            state = env.reset() 
            episod_reward : int = 0
            global_step : int = 1

            # Initialze the step function
            step_function(
                max_steps_per_episode=max_steps_per_episode,
                Q_Network=Q_Network,
                Target_Q_Network=Target_Q_Network,
                optimizers=optimizer,
                state=state,
                episode_reward=episod_reward,
                global_step=global_step
            )

            # print("End of the epidode !!!!")

    except Exception as e:
        print(f"Error occoured in DQN : {e}")


DQN()        

# print the actionmap in relay buffers 
actions_map = ['↑', '→', '↓', '←']

def print_policy(Q):
    for i in range(4):
        row = ''
        for j in range(4):
            s = i * 4 + j
            if (i, j) == (3, 3):
                row += ' G '
            else:
                best_a = np.argmax(Q[s])
                row += f' {actions_map[best_a]} '
        print(row)

# append the values in the reply_buffer and provide the Q values 
for _ in range(len(replay_buffer)):
    value = replay_buffer.pop()
    Q_value = value[-1]
    print("Printing the policy!!!!")
    print_policy(Q_value)