import os
import sys
import gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0,os.path.dirname(CURRENT_PATH))
from envs import Env_V3

params = {
    'is_render': False, 
    'is_good_view': False,
    'is_train' : True,
    'show_boundary' : False,
    'add_moving_obstacle' : False,
    'moving_obstacle_speed' : 0.15,
    'moving_init_direction' : -1,
    'moving_init_axis' : 0,
    'workspace' : [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5],
    'max_steps_one_episode' : 512,
    'num_obstacles' : 1,
    'prob_obstacles' : 0.8,
    'obstacle_box_size' : [0.04,0.04,0.002],
    'obstacle_sphere_radius' : 0.05,
    'obstacle_shape': "BOX"
}

# The noise objects for TD3

def make_env(rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = Env_V3(
            is_render=params['is_render'],
            is_good_view=params['is_good_view'],
            is_train=params['is_train'],
            show_boundary=params['show_boundary'],
            add_moving_obstacle=params['add_moving_obstacle'],
            moving_obstacle_speed=params['moving_obstacle_speed'],
            moving_init_direction=params['moving_init_direction'],
            moving_init_axis=params['moving_init_axis'],
            workspace=params['workspace'],
            max_steps_one_episode=params['max_steps_one_episode'],
            num_obstacles=params['num_obstacles'],
            prob_obstacles=params['prob_obstacles'],
            obstacle_box_size=params['obstacle_box_size'],
            obstacle_sphere_radius=params['obstacle_sphere_radius']
            )
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__=='__main__':

    # Separate evaluation env
    eval_env = Env_V3(
        is_render=params['is_render'],
        is_good_view=params['is_good_view'],
        is_train=False,
        show_boundary=params['show_boundary'],
        add_moving_obstacle=params['add_moving_obstacle'],
        moving_obstacle_speed=params['moving_obstacle_speed'],
        moving_init_direction=params['moving_init_direction'],
        moving_init_axis=params['moving_init_axis'],
        workspace=params['workspace'],
        max_steps_one_episode=params['max_steps_one_episode'],
        num_obstacles=params['num_obstacles'],
        prob_obstacles=params['prob_obstacles'],
        #obstacle_box_size=params['obstacle_box_size'],
        #obstacle_sphere_radius=params['obstacle_sphere_radius']
        )
    eval_env = Monitor(eval_env)
    # load env
    env = SubprocVecEnv([make_env(i) for i in range(8)])
    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=1e8, verbose=1)

    run_name = "RUN_BOX_" + str(0)

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=f'./models/{run_name}/best/',
                       log_path=f'./models/{run_name}/best/', eval_freq=10000,
                       deterministic=True, render=False)
    
    # Save a checkpoint every ? steps
    checkpoint_callback = CheckpointCallback(save_freq=51200, save_path=f'./models/{run_name}/ckp_logs/',
                                        name_prefix='reach')

    # Action Noise
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=np.ones(n_actions))


    # Create the callback list
    callback = CallbackList([checkpoint_callback, callback_max_episodes, eval_callback])
    model = TD3("MultiInputPolicy", env, train_freq=1, learning_rate=0.001, tau=0.001, gamma=0.99, action_noise=action_noise, verbose=1, tensorboard_log='./models/{run_name}/tf_logs/')
    model = TD3.load('./models/RUN_SHPERE/reach_1000000', env=env)
    model.learn(
        total_timesteps=5000,
        n_eval_episodes=64,
        callback=callback)
    model.save(f'./models/{run_name}/reach')
