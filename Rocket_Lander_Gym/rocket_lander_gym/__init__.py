from gym.envs.registration import register


register(
    id='RocketLander-v0',
    entry_point='rocket_lander_gym.envs:RocketLander',
    max_episode_steps=1600,
    reward_threshold=0,
)
