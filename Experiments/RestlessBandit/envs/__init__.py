from gym.envs.registration import register

register(
    id='metacognition-v0',
    entry_point='envs.bandit:MetacognitionEnv',
    kwargs={'num_actions': 2, 'mean_rewards':[40, 60], 'var_rewards':[8, 8], 'step_range_per_block':[18, 22], 'no_blocks':20}
)
