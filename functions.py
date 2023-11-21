import numpy as np
import matplotlib.pyplot as plt


def plot_response(data_dict):
    """plot the responses of the rocket while landing"""
    plt.figure(figsize=(12, 6))
    for key, value in data_dict.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.grid()
    plt.ylim(-1.1, 1.1)
    plt.title('2-DoF PID Control')
    plt.ylabel('Value')
    plt.xlabel('Steps')
    plt.show()