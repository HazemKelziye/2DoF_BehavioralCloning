import gym.spaces
import rocket_lander_gym
from cnn import CNN2D_2headed_3
import torch
from functions import plot_response
from simulation import run_simulation


# load environment
env = gym.make('RocketLander-v0')
# load the pytorch model
model = CNN2D_2headed_3()
model.load_state_dict(torch.load("/home/vboxuser/PycharmProjects/2DoF_PID22/multi_headed_model_4_v1.pth", map_location=torch.device('cpu')))
model.eval()

env.reset()
# run simulation and collect data for plotting
data = run_simulation(env, model)
plot_response(data)
