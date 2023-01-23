import importlib
import stable_baselines3

from typing import Callable
import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

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
    'max_steps_one_episode' : 1024,
    'num_obstacles' : 3,
    'prob_obstacles' : 0.8,
    'obstacle_box_size' : [0.04,0.04,0.002],
    'obstacle_sphere_radius' : 0.04       
}

def make_env(env_class, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """
        def _init() -> gym.Env:
            env = env_class(is_render=params['is_render'],
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

class Config():
    
    def __init__(self, config_path='config.test_config_file'):
        cfg = importlib.import_module(config_path)
        self.env = self.create_environments(cfg.env, cfg.train_cfg['num_of_envs'])
        self.train_cfg = cfg.train_cfg
    
    
    def get_model(self):
        model = getattr(stable_baselines3, self.train_cfg['model'])("MultiInputPolicy", self.env, batch_size=256, verbose=1, tensorboard_log='./models/tf_logs/')
        return model
    
    def create_environments(self, cfg_env, num_of_envs):
        environment_module = importlib.import_module('envs.' + cfg_env['type'])
        env_class = getattr(environment_module, 'Env')
        environments =  SubprocVecEnv([make_env(env_class, i) for i in range(num_of_envs)])
        return environments
    
    
        