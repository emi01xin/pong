## Coding a neural network to teach an AI to play pong through deep learning using OpenAI Gym ##
# Source: http://karpathy.github.io/2016/05/31/rl/
# Code Source: https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

import gym
import numpy as np

def main():
    # use OpenAI Gym to create game envirornment
    env = gym.make("Pong-v0")

    # get initial image of game
    observation = env.reset()

    # hyperparameters
    batch_size = 10 # num of rounds to play before updating weights of network
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 80*80 # dimensions for image
    learning_rate = 1e-4

    prev_processed_observations = None
    running_reward = None
    reward_sum = 0
    episode_number = 0

    # initialize each layer's weights 
    weights = {
        '1' : np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions),
        '2' : np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    }

    # set up initial parameters for RMSProp algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    expectation_g_squared = {}
    g_dict = {}
    for layer_name in weights.keys():
          expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
          g_dict[layer_name] = np.zeros_like(weights[layer_name])

    # collect information to compute gradient based on the result
    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []

    
    # infinite loop where we continually make a move and learn based on the results of the move
    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)

        # send observations through neural net
        hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights)
        
        episode_observations.append(processed_observations)
        episode_hidden_layer_values.append(hidden_layer_values)

        # record results and choose an action
        action = choose_action(up_probability)

        # carry out action
        observation, reward, done, info = env.step(action)

        reward_sum += reward
        episode_rewards.append(reward)

        # loss function reference: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability # gradient per action
        episode_gradient_log_ps.append(loss_function_gradient)

        # if episode is finished
        if done: 
            episode_number += 1
            
            # combine following values for episode
            episode_hidden_layer_values = np.vstack(episode_hidden_layer_values)
            episode_observations = np.vstack(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)

            # tweak gradient of log_ps based on discounted rewards
            episode_gradient_log_ps_discounted = discount_with_rewards(episode_gradient_log_ps, episode_rewards, gamma)
            
            gradient = compute_gradient(
                episode_gradient_log_ps_discounted,
                episode_hidden_layer_values,
                episode_observations,
                weights
            )

            # sum gradient when we hit batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            # upgrade weights every batch episode by applying RMSProp
            if episode_number % batch_size == 0:
                update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)

            # reset values for new episode
            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], [] 
            observation = env.reset()

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            
            print ("episode %d reward total: %f, running mean: %f ...starting new episode" %(episode_number, reward_sum, running_reward))
            
            reward_sum = 0
            prev_processed_observations = None


# preprocess image
def downsample(image):
    return image[::2, ::2, :] # takes every other pixel, halves resolution of image

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def remove_color(image):
    return image[:, :, 0]

def relu(vector):
    vector[vector < 0] = 0
    return vector

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

# convert image given by OpenAI Gym to train neural network
def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):

    # convert 210x160x3 uint8 frame into a 6400 float vector
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1

    # convert from 80x80 matrix to 1600x1 matrix
    processed_observation = processed_observation.astype(np.float).ravel()

    # store just the difference between current frame and prev frame
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation

    else:
        input_observation = np.zeros(input_dimensions)

    prev_processed_observations = processed_observation
            
    return input_observation, prev_processed_observations
            
# based on observation matrix and weights, compute new hidden layer values and new output layer values
def apply_neural_nets(observation_matrix, weights):
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
            
    return hidden_layer_values, output_layer_values
            
# compute gradient for episode
def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
            
    # backpropagation reference: http://neuralnetworksanddeeplearning.com/chap2.html
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()

    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
            
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):

    # reference: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        return 2 # up
    else:
        return 3 # down

# discount gradient with normalized rewards
def discount_with_rewards(gradient_log_ps, episode_rewards, gamma):
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    
    # standardize rewards to be unit normal to help control gradient estimator variance
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_ps * discounted_episode_rewards

# rewards from earlier frames are discounted more than later frames
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

    
main()
            






























































            
