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
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import time

CURRENT_PATH = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(CURRENT_PATH))
from env import Env_Sangiovanni as Env

params = {
    'is_render': True,
    'is_good_view': True,
    'is_train': False,
    'show_boundary': False,
    'add_moving_obstacle': False,
    'moving_obstacle_speed': 0.2,
    'moving_init_direction': 1,
    'moving_init_axis': 0,
    'workspace': [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
    'max_steps_one_episode': 300,
    'num_obstacles': 3,
    'prob_obstacles': 0.8,
    'obstacle_box_size': [0.04, 0.04, 0.002],
    'obstacle_sphere_radius': 0.08
}

TEST_MODE = 2

if __name__ == '__main__':
    initial_time = 1.

    if TEST_MODE == 1:
        t0 = datetime.datetime.now()
        env = Env(
            is_render=params['is_render'],
            is_good_view=params['is_good_view'],
            is_train=params['is_train'],
            show_boundary=params['show_boundary'],
            add_moving_obstacle=False,
            moving_obstacle_speed=params['moving_obstacle_speed'],
            moving_init_direction=params['moving_init_direction'],
            moving_init_axis=params['moving_init_axis'],
            workspace=params['workspace'],
            max_steps_one_episode=params['max_steps_one_episode'],
            num_obstacles=params['num_obstacles'],
            prob_obstacles=params['prob_obstacles'],
            obstacle_box_size=params['obstacle_box_size'],
            obstacle_sphere_radius=params['obstacle_sphere_radius'],
            test_mode=1
        )
        model = TD3.load('./models/RUN_6TD3/best/best_model', env=env)
        t1 = datetime.datetime.now()
        initial_time = (t1 - t0).total_seconds()
        print('time of initialization: ', initial_time)

        for i in range(30):
            reset_time = 0.
            pred_time = 0.
            exec_time = 0.
            done = False
            t0 = datetime.datetime.now()
            obs = env.reset()
            t1 = datetime.datetime.now()
            reset_time = (t1 - t0).total_seconds()
            while not done:
                t0 = datetime.datetime.now()
                action, _states = model.predict(obs)
                #time.sleep(5)
                t1 = datetime.datetime.now()
                pred_time += (t1 - t0).total_seconds()
                t0 = datetime.datetime.now()
                obs, rewards, done, info = env.step(action)
                t1 = datetime.datetime.now()
                exec_time += (t1 - t0).total_seconds()
            print('reset time: ', reset_time)
            print('prediction time for each episode: ', pred_time)
            print('execution time for each episode: ', exec_time)
    if TEST_MODE == 2:
        t0 = datetime.datetime.now()
        env = Env(
            is_render=params['is_render'],
            is_good_view=params['is_good_view'],
            is_train=params['is_train'],
            show_boundary=params['show_boundary'],
            add_moving_obstacle=True,
            moving_obstacle_speed=params['moving_obstacle_speed'],
            moving_init_direction=params['moving_init_direction'],
            moving_init_axis=params['moving_init_axis'],
            workspace=params['workspace'],
            max_steps_one_episode=params['max_steps_one_episode'],
            num_obstacles=params['num_obstacles'],
            prob_obstacles=params['prob_obstacles'],
            obstacle_box_size=params['obstacle_box_size'],
            obstacle_sphere_radius=params['obstacle_sphere_radius'],
            test_mode=2
        )
        model = TD3.load('./models/RUN_6TD3/best/best_model', env=env)
        t1 = datetime.datetime.now()
        initial_time = (t1 - t0).total_seconds()
        print('time of initialization: ', initial_time)
        for i in range(30):
            reset_time = 0.
            pred_time = 0.
            exec_time = 0.
            done = False
            t0 = datetime.datetime.now()
            obs = env.reset()
            t1 = datetime.datetime.now()
            reset_time = (t1 - t0).total_seconds()
            while not done:
                t0 = datetime.datetime.now()
                action, _states = model.predict(obs)
                time.sleep(0.2)
                t1 = datetime.datetime.now()
                pred_time += (t1 - t0).total_seconds()
                t0 = datetime.datetime.now()
                obs, rewards, done, info = env.step(action)
                t1 = datetime.datetime.now()
                exec_time += (t1 - t0).total_seconds()
            print('reset time: ', reset_time)
            print('prediction time for each episode: ', pred_time)
            print('execution time for each episode: ', exec_time)
    if TEST_MODE == 3:
        t0 = datetime.datetime.now()
        env = Env(
            is_render=params['is_render'],
            is_good_view=params['is_good_view'],
            is_train=params['is_train'],
            show_boundary=params['show_boundary'],
            add_moving_obstacle=False,
            moving_obstacle_speed=params['moving_obstacle_speed'],
            moving_init_direction=params['moving_init_direction'],
            moving_init_axis=params['moving_init_axis'],
            workspace=params['workspace'],
            max_steps_one_episode=params['max_steps_one_episode'],
            num_obstacles=params['num_obstacles'],
            prob_obstacles=params['prob_obstacles'],
            obstacle_box_size=params['obstacle_box_size'],
            obstacle_sphere_radius=params['obstacle_sphere_radius'],
            test_mode=3,
            init_pos=[0.25, 0.4, 0.3],
            target=[0, 0.4, 0.25]
        )
        model = PPO.load('./models/ckps/model1', env=env)
        t1 = datetime.datetime.now()
        initial_time = (t1 - t0).total_seconds()
        print('time of initialization: ', initial_time)
        for i in range(30):
            reset_time = 0.
            pred_time = 0.
            exec_time = 0.
            done = False
            t0 = datetime.datetime.now()
            obs = env.reset()
            t1 = datetime.datetime.now()
            reset_time = (t1 - t0).total_seconds()
            while not done:
                t0 = datetime.datetime.now()
                action, _states = model.predict(obs)
                time.sleep(1)
                t1 = datetime.datetime.now()
                pred_time += (t1 - t0).total_seconds()
                t0 = datetime.datetime.now()
                obs, rewards, done, info = env.step(action)
                t1 = datetime.datetime.now()
                exec_time += (t1 - t0).total_seconds()
            obs = env._update_target(target=[-0.25, 0.4, 0.25])
            done = False
            while not done:
                t0 = datetime.datetime.now()
                action, _states = model.predict(obs)
                time.sleep(1)
                t1 = datetime.datetime.now()
                pred_time += (t1 - t0).total_seconds()
                t0 = datetime.datetime.now()
                obs, rewards, done, info = env.step(action)
                t1 = datetime.datetime.now()
                exec_time += (t1 - t0).total_seconds()
            print('reset time: ', reset_time)
            print('prediction time for each episode: ', pred_time)
            print('execution time for each episode: ', exec_time)


