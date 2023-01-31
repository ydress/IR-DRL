import sys
import os
import numpy as np
import pybullet as p
import gym
from gym import spaces
from typing import Tuple, List
import time
import pandas
import pybullet_data
from random import choice
from numpy import newaxis as na
import copy
from collections import deque
from pybullet_util import MotionExecute
from math_util import euler_from_quaternion
from sklearn.metrics.pairwise import euclidean_distances

CURRENT_PATH = os.path.abspath(__file__)
BASE = os.path.dirname(os.path.dirname(CURRENT_PATH))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.dirname(CURRENT_PATH))

class Env_V3(gym.Env):
    """
    Differences to Env_V2:
    - interpret actions as velocities and control the robot using velocity control; max speed is 10; action space is
    normalized, the actions are later multiplied by 10 to reflect the real range
    - removed shaking reward
    - increased limits of joint positions to allow max joint configurations
    - added joint angular velocities back to obs space
    - updated limits in observations space to enclose the observations more tightly
    - removed end_effector_orn from observation space
    - change reward for distance to target from huber loss to linear function
    - changed reward hyperparameters
    """

    # TODO:

    def __init__(
            self,
            is_render: bool = False,
            is_good_view: bool = False,
            is_train: bool = True,
            show_boundary: bool = True,
            add_moving_obstacle: bool = False,
            moving_obstacle_speed: float = 0.15,
            moving_init_direction: int = -1,
            moving_init_axis: int = 0,
            workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5],
            max_steps_one_episode: int = 1024,
            num_obstacles: int = 1,
            prob_obstacles: float = 0.8,
            obstacle_box_size: list = [0.002, 0.1, 0.06],
            obstacle_sphere_radius: float = 0.06,
            obstacle_shape = "BOX",
            test_mode: int = 0,

    ):
        '''
        is_render: start GUI
        is_good_view: slow down the motion to have a better look
        is_tarin: training or testing
        '''
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.is_train = is_train
        self.DISPLAY_BOUNDARY = show_boundary
        self.extra_obst = add_moving_obstacle
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # test mode
        self.test_mode = test_mode
        # set the area of the workspace
        self.x_low_obs = workspace[0]
        self.x_high_obs = workspace[1]
        self.y_low_obs = workspace[2]
        self.y_high_obs = workspace[3]
        self.z_low_obs = workspace[4]
        self.z_high_obs = workspace[5]

        # for the moving
        self.direction = moving_init_direction
        self.moving_xy = moving_init_axis  # 0 for x, 1 for y
        self.moving_obstacle_speed = moving_obstacle_speed

        # action sapce
        self.action = np.zeros(6, dtype=np.float32)
        self.previous_action = np.zeros(6, dtype=np.float32)
        # normalize action space from -1 to 1 and alter multiply the action by 10:
        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # set joint speed; in order to make Bennos step function work
        self.joint_speed = 0.015
        self.joints_lower_limits = np.array([-3.228858, -3.228858, -2.408553, -6.108651, -2.26891, -6.108651])
        self.joints_upper_limits = np.array([3.22885911, 1.13446401, 3.0543261, 6.10865238, 2.26892802, 6.1086523])
        self.joints_range = self.joints_upper_limits - self.joints_lower_limits

        # parameters for spatial infomation
        self.home = [0, np.pi / 2, -np.pi / 6, -2 * np.pi / 3, -4 * np.pi / 9, np.pi / 2, 0.0]
        self.target_position = None
        self.obsts = []
        self.vel_checker = 0
        self.past_distance = deque([])

        # observation space
        """ According to paper 2018
        TODO: 
            * Joint positions
            * Joint Velocities
            * Traget point position
            * EE Position (Assumed to be known)
            * Obstacle position (Assumed to be known)
            * velocity (Assumed to be correctly estimated)
        """
        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_angular_velocities = np.zeros(6, dtype=np.float32)
        self.target_position = np.zeros(3, dtype=np.float32)
        self.end_effector_position = np.zeros(3, dtype=np.float32)
        self.obstacle_position = np.zeros(9, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)

        self.robot_skeleton = np.zeros((10, 3), dtype=np.float32)

        obs_spaces = {
            "joint_positions": spaces.Box(low=7, high=7, shape=(6,)),
            "joint_angular_velocities": spaces.Box(low=-10, high=10, shape=(6,)),
            "target_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "end_effector_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "obstacle_position": spaces.Box(low=-2, high=2, shape=(9, )),
            "obstacle_velocity": spaces.Box(low=-10, high=10, shape=(3,))
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # step counter
        self.step_counter = 0
        # max steps in one episode
        self.max_steps_one_episode = max_steps_one_episode
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = '../ur5_description/urdf/ur5.urdf'
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        # obstacles
        self.num_obstacles = num_obstacles
        self.prob_obstacles = prob_obstacles
        self.obstacle_box_size = obstacle_box_size
        self.obstacle_radius = obstacle_sphere_radius
        self.obstacle_shape = obstacle_shape

        # # for debugging camera
        # self.x_offset = p.addUserDebugParameter("x", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("y", -2, 2, 0)
        # self.z_offset = p.addUserDebugParameter("z", -2, 2, 0)
        # set image width and height

        # parameters of augmented targets for training
        if self.is_train:
            self.distance_threshold = 0.01
            self.distance_threshold_last = 0.01
            self.distance_threshold_increment_p = 0.0001
            self.distance_threshold_increment_m = 0.001
            self.distance_threshold_max = 0.01
            self.distance_threshold_min = 0.01
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.01
            self.distance_threshold_last = 0.01
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.01
            self.distance_threshold_min = 0.01

        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0

    def _set_robot_skeleton(self):
        robot_skeleton = []
        for i in range(p.getNumJoints(self.RobotUid)):
            if i > 2:
                if i == 3:
                    robot_skeleton.append(p.getLinkState(self.RobotUid, i)[0])
                    robot_skeleton.append(p.getLinkState(self.RobotUid, i)[4])
                else:
                    robot_skeleton.append(p.getLinkState(self.RobotUid, i)[0])
        self.robot_skeleton = np.asarray(robot_skeleton, dtype=np.float32).round(10)
        # add 3 additional points along the arm
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[1] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[2] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[6] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        # add an additional point on the right side of the head
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        (self.robot_skeleton[3] - 1.5 * (
                                                    self.robot_skeleton[3] - self.robot_skeleton[2]))[na, :], axis=0)

    def _set_home(self):

        rand = np.float32(np.random.rand(3, ))
        init_x = (self.x_low_obs + self.x_high_obs) / 2 + 0.5 * (rand[0] - 0.5) * (self.x_high_obs - self.x_low_obs)
        init_y = (self.y_low_obs + self.y_high_obs) / 2 + 0.5 * (rand[1] - 0.5) * (self.y_high_obs - self.y_low_obs)
        init_z = (self.z_low_obs + self.z_high_obs) / 2 + 0.5 * (rand[2] - 0.5) * (self.z_high_obs - self.z_low_obs)
        init_home = [init_x, init_y, init_z]

        rand_orn = np.float32(np.random.uniform(low=-np.pi, high=np.pi, size=(3,)))
        init_orn = np.array([np.pi, 0, np.pi] + 0.1 * rand_orn)
        return init_home, init_orn


    def _create_visual_box(self, halfExtents):
        visual_id = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _create_collision_box(self, halfExtents):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=halfExtents)
        return collision_id

    def _create_visual_sphere(self, radius):
        visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _create_collision_sphere(self, radius):
        collision_id = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius)
        return collision_id

    def _set_target_position(self):
        val = False
        while not val:
            target_x = np.random.uniform(self.x_low_obs + 0.04, self.x_high_obs - 0.04)
            target_y = np.random.uniform(self.y_low_obs, self.y_high_obs - 0.11)
            target_z = np.random.uniform(self.z_low_obs, self.z_high_obs)
            target_position = [target_x, target_y, target_z]
            if (np.linalg.norm(np.array(self.init_home) - np.array(target_position),
                               None) > 0.3):  # so there is no points the robot cant reach
                val = True

        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        return target_position

    def _add_obstacles(self):
        obsts = []
        for item in range(self.num_obstacles):
            position = 0.5 * (np.array(self.init_home) + np.array(self.target_position)) + 0.05 * np.random.uniform(
                low=-1, high=1, size=(3,))
            if self.obstacle_shape == "BOX":
                obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box(halfExtents=self.obstacle_box_size),
                        baseCollisionShapeIndex=self._create_collision_box(halfExtents=self.obstacle_box_size),
                        basePosition=position
                    )
            else:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_sphere(radius=self.obstacle_radius),
                    baseCollisionShapeIndex=self._create_collision_sphere(radius=self.obstacle_radius),
                    basePosition=position
                )
            obsts.append(obst_id)

        return obsts        

    def _add_moving_plate(self):
        pos = copy.copy(self.target_position)
        if self.moving_xy == 0:
            pos[0] = self.x_high_obs - np.random.random() * (self.x_high_obs - self.x_low_obs)
        if self.moving_xy == 1:
            pos[1] = self.y_high_obs - np.random.random() * (self.y_high_obs - self.y_low_obs)

        pos[2] += 0.05
        obst_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.002]),
            baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.002]),
            basePosition=pos
        )
        return obst_id

    def reset(self):
        p.resetSimulation()

        self.action = None
        self.init_home, self.init_orn = self._set_home()
        self.target_position = self._set_target_position()
        self.obsts = self._add_obstacles()
        self.target_position = np.asarray(self.target_position, dtype=np.float32)

        # reset
        self.step_counter = 0
        self.collided = False
        self.terminated = False
        self.ep_reward = 0
        p.setGravity(0, 0, 0)

        # display boundary
        if self.DISPLAY_BOUNDARY:
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_low_obs])
            p.addUserDebugLine(lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_low_obs],
                               lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_low_obs])

        # load the robot arm
        baseorn = p.getQuaternionFromEuler([0, 0, 0])
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[0.0, -0.12, 0.5], baseOrientation=baseorn,
                                   useFixedBase=True)
        self.motionexec = MotionExecute(self.RobotUid, self.base_link, self.effector_link)
        # time.sleep(5)
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.7 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.7 and self.distance_threshold > self.distance_threshold_min:
                self.distance_threshold -= self.distance_threshold_increment_m
            elif success_rate == 1 and self.distance_threshold == self.distance_threshold_min:
                pass
            else:
                self.distance_threshold = self.distance_threshold_last
            if self.distance_threshold <= self.distance_threshold_min:
                self.distance_threshold = self.distance_threshold_min
            print('current distance threshold: ', self.distance_threshold)
            print("previous distance threshold: ", self.distance_threshold_last)

        # do this step in pybullet
        p.stepSimulation()

        # if the simulation starts with the robot arm being inside an obstacle, reset again
        if p.getContactPoints(self.RobotUid) != ():
            self.reset()

        # set observations
        self._set_obs()

        return self._get_obs()

    def step(self, action):
        self.action = action * 10
        # self.action = np.repeat(10, 6)
        p.setJointMotorControlArray(self.RobotUid,
                                    [1, 2, 3, 4, 5, 6],
                                    p.VELOCITY_CONTROL,
                                    targetVelocities=self.action,
                                    forces=np.repeat(300, 6))

        # perform step

        p.stepSimulation()

        # check collision
        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])
            if len(contacts) > 0:
                self.collided = True

        if self.is_good_view:
            time.sleep(0.02)

        self.step_counter += 1
        # input("Press ENTER")
        return self._reward()

    def _reward(self):
        reward = 0

        # set parameters
        lambda_1 = 1000
        lambda_2 = 100
        lambda_3 = 60
        k = 8
        d_ref = 0.05
        dirac = 0.1

        # set observations
        self._set_obs()
        # print(self.joint_angular_velocities)
        # reward for distance to target
        # TODO: Huber-loss
        
        self.distance = np.linalg.norm(self.end_effector_position - self.target_position, ord=None)
        
        # calculating Huber loss for distance of end effector to target
        if abs(self.distance) < dirac:
            R_E_T = 0.5 * (self.distance ** 2)
        else:
            R_E_T = dirac * (self.distance - 0.5 * dirac)
        R_E_T = -R_E_T

        # reward for distance to obstacle
        # TODO: Calculate distance to obstacle
        # 0 if distance = 1 bad. 0 if far away good
        R_R_O = (d_ref / (self.distances_to_obstacles.min() + d_ref)) ** k

        R_R_O = -R_R_O

        # calculate motion size
        R_A = - np.sum(np.square((self.action / 10)))

        # calculate reward
        reward += lambda_1 * R_E_T + lambda_2 * R_R_O + lambda_3 * R_A

        # check if out of boundery
        x = self.end_effector_position[0]
        y = self.end_effector_position[1]
        z = self.end_effector_position[2]
        d = 0.1
        out = bool(
            x < self.x_low_obs - d
            or x > self.x_high_obs + d
            or y < self.y_low_obs - d
            or y > self.y_high_obs + d
            or z < self.z_low_obs - d
            or z > self.z_high_obs + d)

        # success
        is_success = False
        if self.distance < self.distance_threshold:
            self.terminated = True
            is_success = True
            self.success_counter += 1
            #reward += 500
        #elif self.collided:
            #self.terminated = True
            #reward += -500
        elif self.step_counter >= self.max_steps_one_episode:
            #reward += -100
            self.terminated = True
        #elif out:
            #self.terminated = True
            #reward -= 500
            

        self.ep_reward += reward

        info = {'step': self.step_counter,
                "n_episode": self.episode_counter,
                "success_counter": self.success_counter,
                'distance': self.distance,
                "min_distance_to_obstacles": self.distances_to_obstacles.min(),
                "reward_1": lambda_1 * R_E_T,
                "reward_2": lambda_2 * R_R_O,
                "reward_3": lambda_3 * R_A,
                'reward': reward,
                "ep_reward": self.ep_reward,
                'collided': self.collided,
                'is_success': is_success,
                'out': out
                }

        if self.terminated:
            print(info)
            # logger.debug(info)
        return self._get_obs(), reward, self.terminated, info

    def _set_obs(self):
        """
        Collect observetions for observation space.
        
            * Joint positions
            * Joint Velocities
            * Traget point position
            * EE Position (Assumed to be known)
            * Obstacle position (Assumed to be known)
            * Obstacle velocity (Assumed to be correctly estimated)
        
        """
        # set joint positions and angulare velocities
        self.prev_joint_positions = self.joint_positions
        joint_positions = []
        joint_angular_velocities = []
        for i in range(p.getNumJoints(self.RobotUid)):
            joint_positions.append(p.getJointState(self.RobotUid, i)[0])
            joint_angular_velocities.append(p.getJointState(self.RobotUid, i)[1])
        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_positions = self.joint_positions[1:7]  # remove fixed joints
        self.joint_angular_velocities = np.asarray(joint_angular_velocities, dtype=np.float32)
        self.joint_angular_velocities = self.joint_angular_velocities[1:7]  # remove fixed joints

        # set end effector position
        self.end_effector_position = np.asarray(p.getLinkState(self.RobotUid, 7)[0], dtype=np.float32)

        self.target_position = self.target_position

        self._set_robot_skeleton()
        self._set_min_distances_of_robot_to_obstacle(self._calculate_box_points(self.obsts))

        # obstacle positions
        obs_pos = [p.getBasePositionAndOrientation(obs_id)[0] for obs_id in self.obsts]
        obs_vel = [p.getBaseVelocity(obs_id)[0] for obs_id in self.obsts]

        #obs_id = self.obsts[0]

        #obs_pos = p.getBasePositionAndOrientation(obs_id)[0]
        #obs_vel = p.getBaseVelocity(obs_id)[0]
        self.obstacle_position = np.array(obs_pos, dtype=np.float32) #Only use position not orientation
        self.obstacle_velocity = np.array(obs_vel, dtype=np.float32)



    """def _get_obstacle_position(self, obs_id):
        obs_pos = p.getBasePositionAndOrientation(obs_id)[0]

        if self.obstacle_shape == "BOX":
            obs_pos = self._calculate_box_points(obs_pos, self.obstacle_box_size)

        return obs_pos"""

    def _calculate_box_points(self, positions, halfextends):
        points = positions

        h = np.array([[-1.0, -1.0, -1.0],
                      [-1.0, -1.0, 1.0],
                      [1.0, -1.0, -1.0],
                      [1.0, -1.0, 1.0],
                      [-1.0, 1.0, -1.0],
                      [-1.0, 1.0, 1.0],
                      [1.0, 1.0, -1.0],
                      [1.0, 1.0, 1.0]]).reshape((8, 3))

        for pos in positions:

            transform = h * halfextends

            points = np.append(points, pos + transform, axis=0)

        return points

    def _set_min_distances_of_robot_to_obstacle(self, obs_positions) -> None:
        """
        Compute the minimal distances from the robot skeleton to the obstacle points. Also determine the points
        that are closest to each point in the skeleton.
        """

        # Compute minimal euclidean distances from the robot skeleton to the obstacle points
        # if self.points.shape[0] == 0:
        #     self.obstacle_points = np.repeat(np.array([0, 0, -2])[na, :], 10, axis=0)
        #     distances_to_obstacles = euclidean_distances(self.robot_skeleton,
        #                                                  np.array([0, 0, -2])[na, :]).min(axis=1).round(10)
        # else:
        if self.obstacle_shape == "BOX":
            distances = euclidean_distances(self.robot_skeleton, obs_positions)
        else:
            distances = euclidean_distances(self.robot_skeleton, obs_positions) - self.obstacle_radius
        # self.obstacle_points = self.points[distances.argmin(axis=1)]
        distances_to_obstacles = abs(distances.min(axis=1).round(10))

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)

        # p.removeAllUserDebugItems()
        # p.addUserDebugPoints(self.obstacle_points, np.tile([255, 0, 0], self.obstacle_points.shape[0]).reshape(self.obstacle_points.shape),
        #                      pointSize=3)
        # time.sleep(20)

    def _get_obs(self):

        return {
            "joint_positions": self.joint_positions,
            "joint_angular_velocities": self.joint_angular_velocities,
            "target_position": self.target_position,
            "end_effector_position": self.end_effector_position,
            "obstacle_position": self.obstacle_position,
            "obstacle_velocity": self.obstacle_velocity
        }

    def _update_target(self, target):
        self.target_position = target
        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=self.target_position,
        )
        self.step_counter = 0
        self.collided = False

        self.terminated = False
        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        # print ('init position & orientation')
        # print(self.current_pos, self.current_orn)
        # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()
        # do this step in pybullet
        p.stepSimulation()

        # input("Press ENTER")

        return self._get_obs()

class MultiObsEnv(Env_V3):

    def __init__(
            self,
            is_render: bool = False,
            is_good_view: bool = False,
            is_train: bool = True,
            show_boundary: bool = True,
            add_moving_obstacle: bool = False,
            moving_obstacle_speed: float = 0.15,
            moving_init_direction: int = -1,
            moving_init_axis: int = 0,
            workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5],
            max_steps_one_episode: int = 1024,
            num_obstacles: int = 1,
            prob_obstacles: float = 0.8,
            obstacle_box_size: list = [0.002, 0.1, 0.06],
            obstacle_sphere_radius: float = 0.06,
            obstacle_shape = "BOX",
            test_mode: int = 0,

    ):
        '''
        is_render: start GUI
        is_good_view: slow down the motion to have a better look
        is_tarin: training or testing
        '''
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.is_train = is_train
        self.DISPLAY_BOUNDARY = show_boundary
        self.extra_obst = add_moving_obstacle
        if self.is_render:
            self.physicsClient = p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        # test mode
        self.test_mode = test_mode
        # set the area of the workspace
        self.x_low_obs = workspace[0]
        self.x_high_obs = workspace[1]
        self.y_low_obs = workspace[2]
        self.y_high_obs = workspace[3]
        self.z_low_obs = workspace[4]
        self.z_high_obs = workspace[5]

        # for the moving
        self.direction = moving_init_direction
        self.moving_xy = moving_init_axis  # 0 for x, 1 for y
        self.moving_obstacle_speed = moving_obstacle_speed

        # action sapce
        self.action = np.zeros(6, dtype=np.float32)
        self.previous_action = np.zeros(6, dtype=np.float32)
        # normalize action space from -1 to 1 and alter multiply the action by 10:
        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

        # set joint speed; in order to make Bennos step function work
        self.joint_speed = 0.015
        self.joints_lower_limits = np.array([-3.228858, -3.228858, -2.408553, -6.108651, -2.26891, -6.108651])
        self.joints_upper_limits = np.array([3.22885911, 1.13446401, 3.0543261, 6.10865238, 2.26892802, 6.1086523])
        self.joints_range = self.joints_upper_limits - self.joints_lower_limits

        # parameters for spatial infomation
        self.home = [0, np.pi / 2, -np.pi / 6, -2 * np.pi / 3, -4 * np.pi / 9, np.pi / 2, 0.0]
        self.target_position = None
        self.obsts = []
        self.vel_checker = 0
        self.past_distance = deque([])

        # observation space
        """ According to paper 2018
        TODO: 
            * Joint positions
            * Joint Velocities
            * Traget point position
            * EE Position (Assumed to be known)
            * Obstacle position (Assumed to be known)
            * velocity (Assumed to be correctly estimated)
        """
        self.joint_positions = np.zeros(6, dtype=np.float32)
        self.joint_angular_velocities = np.zeros(6, dtype=np.float32)
        self.target_position = np.zeros(3, dtype=np.float32)
        self.end_effector_position = np.zeros(3, dtype=np.float32)
        self.obstacle_position = np.zeros((3, 3), dtype=np.float32)
        self.obstacle_velocity = np.zeros((3, 3), dtype=np.float32)

        self.obstacle_position_filled = np.zeros((3, 3), dtype=np.float32)
        self.obstacle_velocity_filled = np.zeros((3, 3), dtype=np.float32)

        self.robot_skeleton = np.zeros((10, 3), dtype=np.float32)

        obs_spaces = {
            "joint_positions": spaces.Box(low=7, high=7, shape=(6,)),
            "joint_angular_velocities": spaces.Box(low=-10, high=10, shape=(6,)),
            "target_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "end_effector_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "obstacle_position": spaces.Box(low=-2, high=2, shape=(3, 3)),
            "obstacle_velocity": spaces.Box(low=-10, high=10, shape=(3, 3))
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # step counter
        self.step_counter = 0
        # max steps in one episode
        self.max_steps_one_episode = max_steps_one_episode
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = '../ur5_description/urdf/ur5.urdf'
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        # obstacles
        self.num_obstacles = num_obstacles
        self.prob_obstacles = prob_obstacles
        self.obstacle_box_size = obstacle_box_size
        self.obstacle_radius = obstacle_sphere_radius
        self.obstacle_shape = obstacle_shape

        # # for debugging camera
        # self.x_offset = p.addUserDebugParameter("x", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("y", -2, 2, 0)
        # self.z_offset = p.addUserDebugParameter("z", -2, 2, 0)
        # set image width and height

        # parameters of augmented targets for training
        if self.is_train:
            self.distance_threshold = 0.01
            self.distance_threshold_last = 0.01
            self.distance_threshold_increment_p = 0.0001
            self.distance_threshold_increment_m = 0.001
            self.distance_threshold_max = 0.01
            self.distance_threshold_min = 0.01
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.01
            self.distance_threshold_last = 0.01
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.01
            self.distance_threshold_min = 0.01

        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0


    def _get_obs(self):

        self.obstacle_position_filled = self.obstacle_position
        self.obstacle_velocity_filled = self.obstacle_velocity

        for i in range(3 - len(self.obstacle_position)):
            #h_inf = np.array([[np.NINF, np.NINF, np.NINF]])
            self.obstacle_position_filled = np.append(self.obstacle_position_filled, [np.zeros(3)], axis=0)
            self.obstacle_velocity_filled = np.append(self.obstacle_velocity_filled, [np.zeros(3)], axis=0)

        return {
            "joint_positions": self.joint_positions,
            "joint_angular_velocities": self.joint_angular_velocities,
            "target_position": self.target_position,
            "end_effector_position": self.end_effector_position,
            "obstacle_position": self.obstacle_position_filled.reshape((3, 3)),
            "obstacle_velocity": self.obstacle_velocity_filled.reshape((3, 3))
        }

    def _calculate_box_points(self, obs_ids):
        if len(obs_ids) > 0:
            points = np.array([p.getBasePositionAndOrientation(obs_ids[0])[0]])

            h = np.array([[-1.0, -1.0, -1.0],
                          [-1.0, -1.0, 1.0],
                          [1.0, -1.0, -1.0],
                          [1.0, -1.0, 1.0],
                          [-1.0, 1.0, -1.0],
                          [-1.0, 1.0, 1.0],
                          [1.0, 1.0, -1.0],
                          [1.0, 1.0, 1.0]]).reshape((8, 3))

        for obs_id in obs_ids:
            pos = p.getBasePositionAndOrientation(obs_id)[0]

            points = np.append(points, [pos], axis=0)

            halfextends = self.hE[obs_id]

            transform = h * halfextends

            points = np.append(points, pos + transform, axis=0)

        else:
            points = np.zeros((3, 3))

        return points

    def _add_obstacles(self):
        obsts = []
        self.hE = {}
        for item in range(3):
            if np.random.random() > 0.3:
                i = choice([0, 1, 2])
                position = 0.5 * (np.array(self.init_home) + np.array(self.target_position)) + 0.05 * np.random.uniform(
                    low=-1, high=1, size=(3,))
                if i == 0:
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.001]),
                        baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.001]),
                        basePosition=position
                    )
                    obsts.append(obst_id)
                    self.hE[obst_id] = [0.05, 0.05, 0.001]
                if i == 1:
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box([0.001, 0.08, 0.06]),
                        baseCollisionShapeIndex=self._create_collision_box([0.001, 0.05, 0.05]),
                        basePosition=position
                    )
                    obsts.append(obst_id)
                    self.hE[obst_id] = [0.001, 0.08, 0.06]
                if i == 2:
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box([0.08, 0.001, 0.06]),
                        baseCollisionShapeIndex=self._create_collision_box([0.05, 0.001, 0.05]),
                        basePosition=position
                    )
                    obsts.append(obst_id)
                    self.hE[obst_id] = [0.08, 0.001, 0.06]


        if len(obsts) == 0:
            i = choice([0, 1, 2])
            position = 0.5 * (np.array(self.init_home) + np.array(self.target_position)) + 0.05 * np.random.uniform(
                low=-1, high=1, size=(3,))
            if i == 0:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.001]),
                    baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.001]),
                    basePosition=position
                )
                obsts.append(obst_id)
                self.hE[obst_id] = [0.05, 0.05, 0.001]
            if i == 1:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_box([0.001, 0.08, 0.06]),
                    baseCollisionShapeIndex=self._create_collision_box([0.001, 0.05, 0.05]),
                    basePosition=position
                )
                obsts.append(obst_id)
                self.hE[obst_id] = [0.001, 0.08, 0.06]
            if i == 2:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_box([0.08, 0.001, 0.06]),
                    baseCollisionShapeIndex=self._create_collision_box([0.05, 0.001, 0.05]),
                    basePosition=position
                )
                obsts.append(obst_id)
                self.hE[obst_id] = [0.08, 0.001, 0.06]

        return obsts
