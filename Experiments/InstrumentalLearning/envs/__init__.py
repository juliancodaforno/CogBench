from gym.envs.registration import register

register(
    id='palminteri2017-v0',
    entry_point='envs.bandits:OptimismBiasTaskPalminteri',
    kwargs={'num_actions': 2, 'reward_scaling': 1, 'max_steps_per_context':24, 'num_contexts':4},
)

register(
    id='twoarmedbandit-v0',
    entry_point='envs.bandits:TwoArmedBanditTask',
    kwargs={'num_actions': 2, 'reward_scaling': 1, 'max_steps_per_context':24, 'num_contexts':4},
)