import gym.spaces
import numpy as np
import rocket_lander_gym
# from pidcontroller import *
import math
import time
import matplotlib.pyplot as plt
from cnn import CNN2D_2headed_3
import torch


EPISODES_NUMBER = 1
SETPOINTS = [0, -1.05, 0]  # Setpoints/desired-points for optimizing the PID controller
largest_reward = 0  # Dummy var for checking whether we landed successfully or not

state_history = []  # history for storing states and accessing them through t to (t+9)

ACTION_X = 0  # !Setting the throttle's gimbal DoF to 0 permanently!
action_y = -0.2
action_theta = -0.2

# map from the predicted class indices' to the actually discrete value
INVERSE_DISCRETE_ACTIONS = {0: -1.0, 1: -0.6, 2: -0.2, 3: 0.2, 4: 0.6, 5: 1.0}

env = gym.make('RocketLander-v0')

# load the pytorch model
model = CNN2D_2headed_3()
model.load_state_dict(torch.load("/home/vboxuser/PycharmProjects/2DoF_PID22/multi_headed_model_4_v1.pth", map_location=torch.device('cpu')))
model.eval()

def discretize_actions(action):
    """discretize given action, according to specified bins_num"""
    bins_num = 6
    min_val = -1.0
    max_val = 1.0
    # bin_width = (max_val - min_val) / bins_num
    intervals = np.linspace(min_val, max_val, bins_num)

    for i in range(bins_num):
        if action <= intervals[i]:
            return intervals[i]

    return intervals[-1] if action >= intervals[-1] else None


env.reset()

action_set = env.action_space.sample()
observation, _, done, _ = env.step(action_set)
t = 0  # used for capturing the last 10 states

while True:
    time.sleep(0.01)
    print()
    print("t-timestep:", t)
    env.render()
    observation, reward, done, _ = env.step(action_set)

    # Taking actions w.r.t. the PID controller's feedback, If one of the legs contacts the ground i.e.
    # action_y = np.clip(y_controller.update(
    #     [observation[1], observation[8]], abs(observation[0]) + SETPOINTS[1]), -1.0, 1.0)

    # action_theta = np.clip(theta_controller.update([observation[2], observation[9]],
    #                                                (math.pi / 4) * (observation[0] + observation[7])), -1, 1)

    # Discretizing the input actions y and theta
    action_set = np.array([ACTION_X, action_y, action_theta])

    state_history.append([list(observation)[0]
                             , list(observation)[1]
                             , list(observation)[2]
                             , list(observation)[7]
                             , list(observation)[8]])

    # check t >= 10 for state capturing from t - 9
    if t >= 10:
        state_0 = state_history[t - 9]
        state_1 = state_history[t - 8]
        state_2 = state_history[t - 7]
        state_3 = state_history[t - 6]
        state_4 = state_history[t - 5]
        state_5 = state_history[t - 4]
        state_6 = state_history[t - 3]
        state_7 = state_history[t - 2]
        state_8 = state_history[t - 1]
        state_9 = state_history[t]

        ten_states = [state_0 + state_1 + state_2 + state_3 + state_4
                      + state_5 + state_6 + state_7 + state_8 + state_9]

        np_ten_states = np.array(ten_states)
        input_states = np.array([arr.reshape(10, 5).T for arr in np_ten_states], dtype=np.float32)
        input_states = torch.from_numpy(input_states)
        input_states = input_states.view(-1, 1, 5, 10)

        with torch.no_grad():
            y1_pred, y2_pred = model(input_states)
            prediction1 = torch.softmax(y1_pred, dim=1)
            prediction2 = torch.softmax(y2_pred, dim=1)
            predicted_class1 = prediction1.argmax(dim=1)
            predicted_class2 = prediction2.argmax(dim=1)
            action_y = INVERSE_DISCRETE_ACTIONS[predicted_class1.item()]
            action_theta = INVERSE_DISCRETE_ACTIONS[predicted_class2.item()]
            print("y1_prediction: ", prediction1)
            print("y1_predicted class: ", predicted_class1)
            print("CNN's action_y : ", action_y)
            print("y2_prediction: ", prediction2)
            print("y2_predicted class: ", predicted_class2)
            print("CNN's action_theta : ", action_theta)

    else:
        action_y = -0.2
        action_theta = -0.2

    t += 1

    # Making a scheme for learning whether the landing was successful or not
    largest_reward = reward if reward > largest_reward else largest_reward
    if done:
        # Deciding whether the landing was successful or not
        success = True if largest_reward >= 0.05 else False
        print(f"Simulation done : {success}.")

        break

env.close()

state_history = []  # empty state_history for storing the new state_history variables, for the new episode


