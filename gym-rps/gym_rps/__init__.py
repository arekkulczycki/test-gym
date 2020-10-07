from gym.envs.registration import register

register(
    id='rps-v0',
    entry_point='gym_rps.envs:RockPaperScissors',
)
