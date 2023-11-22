import numpy as np
import matplotlib.pyplot as plt


def plot_response(data_dict):
    """plot the responses of the rocket while landing"""
    plt.figure(figsize=(8, 5))
    for key, value in data_dict.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.grid()
    plt.ylim(-1.1, 1.1)
    plt.title('2-DoF Behavioral-Cloning Control')
    plt.ylabel('Value')
    plt.xlabel('Steps')
    plt.show()