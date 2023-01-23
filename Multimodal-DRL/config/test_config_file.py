# This is a example file for a config file to load different training environments, Sensors, sensor positions and reward functions from

env = dict(
    type='Static',
    reward='DistanceToTarget',
    max_number_of_obstacles='1',
    sensors={
        1:dict(
            type='RGBD',
            position='front'
        ),
        2:dict(
            type='LIDAR',
            position='top'
        ),
    },
    fusion = 'concat'
)

train_cfg = dict(
    model='PPO',
    num_of_envs=8,
    max_steps=1024,
    total_timesteps=1e10,
    n_eval_episodes=64,
)