import numpy as np
import time
import torch

SETPOINTS = {'y': -1.05, 'theta': 0}


def run_simulation(env, model):
    """ Runs the simulation and collects data for plotting. """

    # map from the predicted class indices' to the actually discrete value
    INVERSE_DISCRETE_ACTIONS = {0: -1.0, 1: -0.6, 2: -0.2, 3: 0.2, 4: 0.6, 5: 1.0}
    t = 0  # used for capturing the last 10 states
    largest_reward = 0  # Dummy variable for checking whether we landed successfully or not
    ACTION_X = 0  # !Setting the throttle's gimbal DoF to 0 permanently!
    action_y = -0.2
    action_theta = -0.2
    data = {'x': [], 'y': [], 'theta': [], 'vx': [], 'vy': [], 'omega': []}
    action_set = [ACTION_X, 0, 0]  # Initial action
    state_history = []  # history for storing states and accessing them through t to (t+9)

    while True:
        time.sleep(0.01)
        env.render()

        observation, reward, done, _ = env.step(action_set)

        # append state variables for plotting
        # 0,1,2,7,8,9 corresponds to the specified variables x,y,theta,vx,vy,omega respectively
        data['x'].append(observation[0])
        data['y'].append(observation[1])
        data['theta'].append(observation[2])
        data['vx'].append(observation[7])
        data['vy'].append(observation[8])
        data['omega'].append(observation[9])

        action_set = np.array([ACTION_X, action_y, action_theta])

        state_history.append([observation[0]
                             , observation[1]
                             , observation[2]
                             , observation[7]
                             , observation[8]])

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
    return data
