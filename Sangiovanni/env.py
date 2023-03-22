import copy
from random import choice

import pybullet as pyb

import numpy as np

import gym
from gym import spaces

from pybullet_util import MotionExecute
import time
from numpy import newaxis as na
from sklearn.metrics.pairwise import euclidean_distances


class Env_Sangiovanni(gym.Env):

    def __init__(self,
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
                 obstacle_box_size: list = [0.04, 0.04, 0.002],
                 obstacle_sphere_radius: float = 0.06,
                 test_mode: int = 0,
                 ):
        self.moving_obstacle_speed = moving_obstacle_speed
        self.ep_reward = 0
        self.terminated = None
        self.obsts = None
        self.target_position = None
        self.init_orn = None
        self.init_home = None
        self.action = None
        self.is_render = is_render
        self.is_good_view = is_good_view
        self.is_train = is_train
        self.test_mode = test_mode

        self.obstacle_radius = obstacle_sphere_radius

        if self.is_render:
            self.physicsClient = pyb.connect(pyb.GUI)
        else:
            pyb.connect(pyb.DIRECT)

        # set the area of the workspace
        self.x_low_obs = workspace[0]
        self.x_high_obs = workspace[1]
        self.y_low_obs = workspace[2]
        self.y_high_obs = workspace[3]
        self.z_low_obs = workspace[4]
        self.z_high_obs = workspace[5]

        self.direction = moving_init_direction
        self.moving_xy = moving_init_axis  # 0 for x, 1 for y
        self.moving_obstacle_speed = moving_obstacle_speed

        # train related
        self.step_counter = 0
        self.distance_threshold = 0.02
        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0
        self.max_steps_one_episode = max_steps_one_episode
        self.collided = False

        # Robot related
        self.urdf_root_path = './ur5_description/urdf/ur5.urdf'
        self.base_link = 1
        self.effector_link = 7

        self._init_action_space()
        self._init_observation_space()

    def reset(self):
        """
        1. reset Simulation
        2. Reset recorder
        1. Load PyBullet with robot
        2. Add Obstacles
        3. Add Target
        """

        self.t1 = time.time()
        pyb.resetSimulation()
        pyb.setGravity(0, 0, 0)

        self.recoder = []
        one_info = []

        self.action = None

        # Reset for Test Modes
        # Set Up home, orn, target and obs based on test mode
        if self.test_mode == 1:
            self.init_home, self.init_orn, self.target_position, self.obsts = self._test_1()
        elif self.test_mode == 2:
            self.init_home, self.init_orn, self.target_position, self.obsts = self._test_2()
        else:
            self.init_home, self.init_orn, self.target_position, self.obsts = self._reset_basics()

        self.step_counter = 0
        self.collided = False
        self.ep_reward = 0
        self.terminated = False

        # Load Robot arm
        self._load_robot()

        self.last_time = time.time()
        one_info.append(self.last_time)

        self.episode_counter += 1

        pyb.stepSimulation()

        if pyb.getContactPoints(self.RobotUid) != ():
            self.reset()

        # set observations
        self._set_obs()

        one_info += list(pyb.getLinkState(self.RobotUid, self.effector_link)[4])

        self.recoder.append(one_info)

        return self._get_obs()

    def step(self, action):

        self.action = action * 10

        pyb.setJointMotorControlArray(self.RobotUid,
                                      [1, 2, 3, 4, 5, 6],
                                      pyb.VELOCITY_CONTROL,
                                      targetVelocities=self.action,
                                      forces=np.repeat(300, 6))

        pyb.stepSimulation()

        if self.test_mode != 1:
            for obs_id in self.obsts:
                self._move_obst(obs_id)

        # check collision
        for i in range(len(self.obsts)):
            contacts = pyb.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])
            if len(contacts) > 0:
                self.collided = True

        if self.is_good_view:
            time.sleep(0.02)

        self.step_counter += 1
        return self._reward()

    def _reward(self):
        one_info = []
        reward = 0

        # set parameters
        lambda_1 = 1000
        lambda_2 = 100
        lambda_3 = 60
        k = 8
        d_ref = self.obstacle_radius + 0.01
        dirac = 0.1

        # set observations
        self._set_obs()

        self.current_time = time.time()

        one_info.append(self.current_time)
        one_info += list(pyb.getLinkState(self.RobotUid, self.effector_link)[4])

        self.distance = np.linalg.norm(self.end_effector_position - self.target_position, ord=None)

        self.recoder.append(one_info)
        self.last_time = self.current_time

        # calculating Huber loss for distance of end effector to target
        if abs(self.distance) < dirac:
            R_E_T = 0.5 * (self.distance ** 2)
        else:
            R_E_T = dirac * (self.distance - 0.5 * dirac)
        R_E_T = -R_E_T

        # reward for distance to obstacle
        # 0 if distance = 1 bad. 0 if far away good
        R_R_O = (d_ref / (self.distances_to_obstacles.min() + d_ref)) ** k

        R_R_O = -R_R_O

        # calculate motion size
        R_A = - np.sum(np.square((self.action / 10)))

        # calculate reward
        reward += lambda_1 * R_E_T + lambda_2 * R_R_O + lambda_3 * R_A

        # success
        is_success = False
        if self.distance < self.distance_threshold:
            self.terminated = True
            if not self.collided:
                is_success = True
                self.success_counter += 1
        elif self.step_counter >= self.max_steps_one_episode:
            self.terminated = True

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
                }

        if self.terminated:
            np.savetxt('./trajectory/testcase' + str(self.test_mode) + '/' + str(self.episode_counter) + '.txt',
                       np.asarray(self.recoder))
            print(info)
            # logger.debug(info)
        return self._get_obs(), reward, self.terminated, info

    def _init_action_space(self):
        self.action = np.zeros(6, dtype=np.float32)
        self.previous_action = np.zeros(6, dtype=np.float32)
        # normalize action space from -1 to 1 and alter multiply the action by 10:
        # https://stable-baselines.readthedocs.io/en/master/guide/rl_tips.html
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)

    def _init_observation_space(self):
        # observation space
        """ According to paper 2018
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
        self.obstacle_position = np.zeros(3, dtype=np.float32)
        self.obstacle_velocity = np.zeros(3, dtype=np.float32)

        self.robot_skeleton = np.zeros((10, 3), dtype=np.float32)

        obs_spaces = {
            "joint_positions": spaces.Box(low=7, high=7, shape=(6,)),
            "joint_angular_velocities": spaces.Box(low=-10, high=10, shape=(6,)),
            "target_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "end_effector_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "obstacle_position": spaces.Box(low=-2, high=2, shape=(3,)),
            "obstacle_velocity": spaces.Box(low=-10, high=10, shape=(3,))
        }
        self.observation_space = spaces.Dict(obs_spaces)

    def _reset_basics(self):
        if self.test_mode == 1:
            return self._test_1()
        elif self.test_mode == 2:
            return self._test_2()
        else:
            home, orn = self._set_home()
            target = self._set_target_position(home)
            obs_ids = self._add_obs()
            return home, orn, target, obs_ids

    def _set_home(self):
        rand = np.float32(np.random.rand(3, ))
        init_x = (self.x_low_obs + self.x_high_obs) / 2 + 0.5 * (rand[0] - 0.5) * (self.x_high_obs - self.x_low_obs)
        init_y = (self.y_low_obs + self.y_high_obs) / 2 + 0.5 * (rand[1] - 0.5) * (self.y_high_obs - self.y_low_obs)
        init_z = (self.z_low_obs + self.z_high_obs) / 2 + 0.5 * (rand[2] - 0.5) * (self.z_high_obs - self.z_low_obs)
        init_home = [init_x, init_y, init_z]

        rand_orn = np.float32(np.random.uniform(low=-np.pi, high=np.pi, size=(3,)))
        init_orn = np.array([np.pi, 0, np.pi] + 0.1 * rand_orn)
        return init_home, init_orn

    def _test_1(self):
        '''
        a simple barrier between init position and target
        '''

        init_home = [0.15, 0.4, 0.3]
        init_orn = [np.pi, 0, np.pi]
        target_position = [-0.15, 0.4, 0.3]
        target = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []
        obst_1 = pyb.createMultiBody(
            baseMass=0,
            # baseVisualShapeIndex=self._create_visual_box([0.002,0.1,0.05]),
            baseCollisionShapeIndex=self._create_collision_box([0.002, 0.05, 0.05]),
            basePosition=[0.0, 0.4, 0.3],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        obsts.append(obst_1)

        self.obstacle_radius = 0.05
        return init_home, init_orn, target_position, obsts

    def _create_collision_box(self, halfExtents):
        collision_id = pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents)
        return collision_id

    def _create_visual_box(self, halfExtents):
        visual_id = pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=halfExtents, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _test_2(self):
        init_home = [0.1, 0.3, 0.33]
        init_orn = [np.pi, 0, np.pi]
        target_position = [-0.3, 0.5, 0.25]
        target = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []

        self.obstacle_radius = 0.05

        self.moving_xy = 1
        obs_id = self._add_moving_plate()
        obsts.append(obs_id)

        return init_home, init_orn, target_position, obsts

    def _set_target_position(self, home):
        val = False
        while not val:
            target_x = np.random.uniform(self.x_low_obs + 0.04, self.x_high_obs - 0.04)
            target_y = np.random.uniform(self.y_low_obs, self.y_high_obs - 0.11)
            target_z = np.random.uniform(self.z_low_obs, self.z_high_obs)
            target_position = [target_x, target_y, target_z]
            if (np.linalg.norm(np.array(home) - np.array(target_position),
                               None) > 0.3):  # so there is no points the robot cant reach
                val = True

        target = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        return target_position

    def _add_moving_plate(self):
        if self.test_mode == 2:
            pos = copy.copy(self.target_position)
            pos[1] = 0.4
            pos[2] += 0.05
            obst_id = pyb.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.002]),
                baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.002]),
                basePosition=pos
            )
        else:
            pos = copy.copy(self.target_position)
            if self.moving_xy == 0:
                pos[0] = self.x_high_obs - np.random.random() * (self.x_high_obs - self.x_low_obs)
            if self.moving_xy == 1:
                pos[1] = self.y_high_obs - np.random.random() * (self.y_high_obs - self.y_low_obs)

            pos[2] += 0.05
            obst_id = pyb.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=self._create_visual_box([0.05, 0.05, 0.002]),
                baseCollisionShapeIndex=self._create_collision_box([0.05, 0.05, 0.002]),
                basePosition=pos
            )
        return obst_id

    # TODO: Change to moving obstacle
    def _add_obs(self):
        self.direction = choice([-1, 1])
        self.moving_xy = choice([0, 1])
        pos = copy.copy(self.target_position)
        if self.moving_xy == 0:
            pos[0] = self.x_high_obs - np.random.random() * (self.x_high_obs - self.x_low_obs)
        if self.moving_xy == 1:
            pos[1] = self.y_high_obs - np.random.random() * (self.y_high_obs - self.y_low_obs)

        pos[2] += self.obstacle_radius + 0.0001
        obst_id = pyb.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._create_visual_sphere(radius=self.obstacle_radius),
            baseCollisionShapeIndex=self._create_collision_sphere(radius=self.obstacle_radius),
            basePosition=pos
        )
        return [obst_id]

    def _create_visual_sphere(self, radius):
        visual_id = pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=radius, rgbaColor=[0.5, 0.5, 0.5, 1])
        return visual_id

    def _create_collision_sphere(self, radius):
        collision_id = pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=radius)
        return collision_id

    def _move_obst(self, obs_id):
        dv = 0.005

        old_barr_pos = np.asarray(pyb.getBasePositionAndOrientation(obs_id)[0])
        barr_pos = np.asarray(pyb.getBasePositionAndOrientation(obs_id)[0])
        if self.moving_xy == 0:
            if barr_pos[0] > self.x_high_obs or barr_pos[0] < self.x_low_obs:
                self.direction = -self.direction
            barr_pos[0] += self.direction * self.moving_obstacle_speed * dv
            pyb.resetBasePositionAndOrientation(obs_id, barr_pos,
                                                pyb.getBasePositionAndOrientation(obs_id)[1])
        if self.moving_xy == 1:
            if barr_pos[1] > self.y_high_obs or barr_pos[1] < self.y_low_obs:
                self.direction = -self.direction
            barr_pos[1] += self.direction * self.moving_obstacle_speed * dv
            pyb.resetBasePositionAndOrientation(obs_id, barr_pos,
                                                pyb.getBasePositionAndOrientation(obs_id)[1])

        self.obstacle_velocity = barr_pos - old_barr_pos
        return

    def _load_robot(self):
        # load the robot arm
        baseorn = pyb.getQuaternionFromEuler([0, 0, 0])
        self.RobotUid = pyb.loadURDF(self.urdf_root_path, basePosition=[0.0, -0.12, 0.5], baseOrientation=baseorn,
                                     useFixedBase=True)
        self.motionexec = MotionExecute(self.RobotUid, self.base_link, self.effector_link)
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

    def _set_obs(self):
        """
        Collect observetions for observation space
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
        for i in range(pyb.getNumJoints(self.RobotUid)):
            joint_positions.append(pyb.getJointState(self.RobotUid, i)[0])
            joint_angular_velocities.append(pyb.getJointState(self.RobotUid, i)[1])
        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_positions = self.joint_positions[1:7]  # remove fixed joints
        self.joint_angular_velocities = np.asarray(joint_angular_velocities, dtype=np.float32)
        self.joint_angular_velocities = self.joint_angular_velocities[1:7]  # remove fixed joints

        # set end effector position
        self.end_effector_position = np.asarray(pyb.getLinkState(self.RobotUid, 7)[0], dtype=np.float32)

        self.target_position = self.target_position

        self._set_robot_skeleton()

        obs_id = self.obsts[0]

        obs_pos = pyb.getBasePositionAndOrientation(obs_id)[0]
        #obs_vel = pyb.getBaseVelocity(obs_id)[0]
        self.obstacle_position = np.array(obs_pos, dtype=np.float32)  # Only use position not orientation
        self.obstacle_velocity = np.array(self.obstacle_velocity, dtype=np.float32)

        self._set_min_distances_of_robot_to_obstacle()

    def _get_obs(self):
        return {
            "joint_positions": self.joint_positions,
            "joint_angular_velocities": self.joint_angular_velocities,
            "target_position": self.target_position,
            "end_effector_position": self.end_effector_position,
            "obstacle_position": self.obstacle_position,
            "obstacle_velocity": self.obstacle_velocity
        }

    def _set_robot_skeleton(self):
        robot_skeleton = []
        for i in range(pyb.getNumJoints(self.RobotUid)):
            if i > 2:
                if i == 3:
                    robot_skeleton.append(pyb.getLinkState(self.RobotUid, i)[0])
                    robot_skeleton.append(pyb.getLinkState(self.RobotUid, i)[4])
                else:
                    robot_skeleton.append(pyb.getLinkState(self.RobotUid, i)[0])
        self.robot_skeleton = np.asarray(robot_skeleton, dtype=np.float32).round(10)
        # add 3 additional points along the arm
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[2] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        ((self.robot_skeleton[6] + self.robot_skeleton[0]) / 2)[na, :], axis=0)
        # add an additional point on the right side of the head
        self.robot_skeleton = np.append(self.robot_skeleton,
                                        (self.robot_skeleton[3] - 1.5 * (
                                                self.robot_skeleton[3] - self.robot_skeleton[2]))[na, :], axis=0)

    def _set_min_distances_of_robot_to_obstacle(self) -> None:
        """
        Compute the minimal distances from the robot skeleton to the obstacle points. Also determine the points
        that are closest to each point in the skeleton.
        """

        distances = euclidean_distances(self.robot_skeleton, [self.obstacle_position]) - self.obstacle_radius
        distances_to_obstacles = abs(distances.min(axis=1).round(10))

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)
