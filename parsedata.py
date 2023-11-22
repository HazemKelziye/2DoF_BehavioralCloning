import torch
import numpy as np
import json
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)

ACTIONS_LABEL_ENCODING = {-1.0: 0, -0.6: 1, -0.2: 2, 0.2: 3, 0.6: 4, 1.0: 5}


def parse_data(file_path):
    """
    parse the data from the (St, At) pairs to ([St, St+1, St+2, St+3, ..., St+9], At+9)
    from the json file. and them up together
    :return: S, A1, A2
    """
    temp_s, temp_a1, temp_a2 = [], [], []
    parse_dataset_s, parse_dataset_a1, parse_dataset_a2 = [], [], []

    with open(file_path) as input_file:
        data = json.load(input_file)
        for i in range(1, 1001):
            episode = data[f"episode{i}"]
            if len(episode) >= 10:
                for t in range(len(episode) - 10):
                    state_0, action_0 = episode[t]
                    state_1, action_1 = episode[t + 1]
                    state_2, action_2 = episode[t + 2]
                    state_3, action_3 = episode[t + 3]
                    state_4, action_4 = episode[t + 4]
                    state_5, action_5 = episode[t + 5]
                    state_6, action_6 = episode[t + 6]
                    state_7, action_7 = episode[t + 7]
                    state_8, action_8 = episode[t + 8]
                    state_9, action_9 = episode[t + 9]

                    temp_s += [state_0 + state_1 + state_2 + state_3 + state_4
                               + state_5 + state_6 + state_7 + state_8 + state_9]

                    temp_a1.append(action_9[0])
                    temp_a2.append(action_9[1])

            parse_dataset_s += temp_s
            parse_dataset_a1 += temp_a1
            parse_dataset_a2 += temp_a2
            temp_s, temp_a1, temp_a2 = [], [], []  # empty temporary 10 states and action's list

    return parse_dataset_s, parse_dataset_a1, parse_dataset_a2


def pipeline(file_path):
    states, actions1, actions2 = parse_data(file_path)

    trunc_actions1 = np.round(actions1, 2)
    trunc_actions2 = np.round(actions2, 2)

    actions1_label_enc = np.array([ACTIONS_LABEL_ENCODING[i] for i in trunc_actions1],
                                  dtype=np.float32)
    actions2_label_enc = np.array([ACTIONS_LABEL_ENCODING[i] for i in trunc_actions2],
                                  dtype=np.float32)

    np_X = np.array(states)
    reshaped_X = np.array([arr.reshape(10, 5).T for arr in np_X], dtype=np.float32)

    X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(reshaped_X,
                                                                             actions1_label_enc,
                                                                             actions2_label_enc,
                                                                             train_size=0.98,
                                                                             random=42)
    X_train = torch.from_numpy(X_train)
    X_test = torch.from_numpy(X_test)
    y1_train = torch.from_numpy(y1_train)
    y1_test = torch.from_numpy(y1_test)
    y2_train = torch.from_numpy(y2_train)
    y2_test = torch.from_numpy(y2_test)

    #reshape data to process via CNN2D model
    X_train = X_train.view(-1, 1, 5, 10)
    X_test = X_test.view(-1, 1, 5, 10)

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y1_train = y1_train.to(device)
    y1_test = y1_test.to(device)
    y2_train = y2_train.to(device)
    y2_test = y2_test.to(device)

    return X_train, X_test, y1_train, y1_test, y2_train, y2_test
