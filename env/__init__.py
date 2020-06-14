from gym.envs.registration import register

register(
    id='SimpleHighway-v0',
    entry_point='env.simple_highway:simple_highway_env',
)
