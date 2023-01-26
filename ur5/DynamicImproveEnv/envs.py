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


class Env_V1(gym.Env):
    """
    This gym environment is based on the "DynamicEnv" from the repository https://github.com/ignc-research/IR-DRL.
    The key differences from the base env are the following:
    - lidar sensor was deactivated
    - a RGB-D sensor was added; the sensor takes an image of the current scene in the simulation. The image is then
    transformed into a point cloud. The points for the robot arm, the target and the background are removed during
    pre-processing. What remains are the points for the obstacles, which are used to compute the minimal distance
    from the robot arm skeleton to these points. This metric is then used to further sanction the agent.
    - the following term was added to the reward function: reward += -1 * np.exp(-15 * self.distances_to_obstacles.min())
    - the robot skeleton and the minimal distances of each point in the skeleton to the obstacle points are added
    to the observation space.
    During training and testing we observed the following:
    - due to the point cloud processing, the training duration increased drastically. As the computation and the processing
    of the point cloud is done during each timestep. This can be improved by only performing the computation after the
    simulation is reset. However, this is might only possible when working with environments without moving objects.

    During training and testing the following was observed:
    - The success rate and mean reward started to converge after 17MM timesteps
    - The smoother success rate converged at a value of around 0.8
    - It is important to note that the base env "DynamicEnv" from the repository produces relatively easy navigation
    problems. A lot of the time the obstacles are not even placed between the robot and the target point.
    In cases where the obstacle is in the way, a collision is registered relatively often.
    - The model seems to perform very well in terms of reaching the target point in a smooth and quick manner with
    little to no back and forth shaking movements. However, its ability to avoid obstacles is rather poor.
    """

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
            workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.4],
            max_steps_one_episode: int = 1024,
            num_obstacles: int = 0,  # 3,
            prob_obstacles: float = 0.8,
            obstacle_box_size: list = [0.04, 0.04, 0.002],
            obstacle_sphere_radius: float = 0.04
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
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
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
        self.action = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)  # angular velocities

        # set DataFrame for transforming segmentation mask to an RGB image
        translation_dic_red = pandas.DataFrame.from_dict(
            {-1: 230, 0: 60, 1: 255, 2: 0, 3: 245, 4: 145, 5: 70, 6: 240, 7: 210, 8: 250, 9: 0,
             10: 220, 11: 170}, orient="index", columns=["r"])
        translation_dic_green = pandas.DataFrame.from_dict(
            {-1: 25, 0: 180, 1: 225, 2: 130, 3: 130, 4: 30, 5: 240, 6: 50, 7: 245, 8: 190, 9: 128,
             10: 190, 11: 110}, orient="index", columns=["g"])
        translation_dic_blue = pandas.DataFrame.from_dict(
            {-1: 75, 0: 75, 1: 25, 2: 200, 3: 48, 4: 180, 5: 240, 6: 230, 7: 60, 8: 212, 9: 128,
             10: 255, 11: 40}, orient="index", columns=["b"])
        self.df_seg_to_rgb = pandas.concat([translation_dic_red, translation_dic_green, translation_dic_blue], axis=1)

        # parameters for spatial infomation
        self.home = [0, np.pi / 2, -np.pi / 6, -2 * np.pi / 3, -4 * np.pi / 9, np.pi / 2, 0.0]
        self.target_position = None
        self.obsts = None
        self.current_pos = None
        self.current_orn = None
        self.current_joint_position = None
        self.vel_checker = 0
        self.past_distance = deque([])

        # observation space
        self.state = np.zeros((14,), dtype=np.float32)
        # self.obs_rays = np.zeros(shape=(129,), dtype=np.float32)
        # self.indicator = np.zeros((10,), dtype=np.int8)
        self.distances_to_obstacles = np.zeros((8,), dtype=np.float32)
        self.link_center_of_mass_coordinates = np.zeros((8, 3), dtype=np.float32)
        obs_spaces = {
            'position': spaces.Box(low=-2, high=2, shape=(14,), dtype=np.float32),
            # 'indicator': spaces.Box(low=0, high=2, shape=(10,), dtype=np.int8),
            "distances_to_obstacles": spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32),
            "link_center_of_mass_coordinates": spaces.Box(low=-100, high=100, shape=(8, 3), dtype=np.float32)
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # step counter
        self.step_counter = 0
        # max steps in one episode
        self.max_steps_one_episode = max_steps_one_episode
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = './ur5_description/urdf/ur5.urdf'
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        # obstacles
        self.num_obstacles = num_obstacles
        self.prob_obstacles = prob_obstacles
        self.obstacle_box_size = obstacle_box_size
        self.obstacle_sphere_radius = obstacle_sphere_radius

        # set image width and height
        self._set_camera_matrices()
        self.img_width = 176
        self.img_height = 120

        # parameters of augmented targets for training
        if self.is_train:
            self.distance_threshold = 0.01
            self.distance_threshold_last = 0.01
            self.distance_threshold_increment_p = 0  # 0.001
            self.distance_threshold_increment_m = 0.001
            self.distance_threshold_max = 0.01
            self.distance_threshold_min = 0.000001
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.03
            self.distance_threshold_last = 0.03
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.03
            self.distance_threshold_min = 0.03

        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0

        # # for camera debugging
        # self.x_offset = p.addUserDebugParameter("_x_offset", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("_y_offset", -2, 2, 0.4)
        # self.z_offset = p.addUserDebugParameter("_z_offset", -2, 2, 0.5)

    def _get_image(self) -> Tuple[List[float], List[int]]:
        """
        Retrieve a depth image and a segmentation mask from the camera sensor.
        :return: The depth image and the segmentation mask
        :rtype: Tuple[List[float], List[int]]
        """
        depthImg, segImg = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.camera_view_matrix,
            projectionMatrix=self.camera_proj_matrix)[3:]

        return depthImg, segImg

    def _depth_img_to_point_cloud(self, depth: np.array) -> np.array:
        """
        Compute a point cloud from a given depth image. The computation is done according to this stackoverflow post:
        https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        :param depth: input depth image;
        the amount of points in the image should equal the product of the camera sensors pixel width and height
        :type depth: np.array
        :return: The point cloud in the shape [width x height, 3]
        :rtype: np.array
        """
        # set width and height
        W = np.arange(0, self.img_width)
        H = np.arange(0, self.img_height)

        # compute pixel coordinates
        X = ((2 * W - self.img_width) / self.img_width)[na, :].repeat(self.img_height, axis=0).flatten()[:, na]
        Y = (-1 * (2 * H - self.img_height) / self.img_height)[:, na].repeat(self.img_width, axis=1).flatten()[:, na]
        Z = (2 * depth - 1).flatten()[:, na]

        # transform pixel coordinates into real world ones
        num_of_pixels = self.img_width * self.img_height
        PixPos = np.concatenate([X, Y, Z, np.ones(num_of_pixels)[:, na]], axis=1)
        points = np.tensordot(self.tran_pix_world, PixPos, axes=(1, 1)).swapaxes(0, 1)
        points = (points / points[:, 3][:, na])[:, 0:3]

        return points

    def _prepreprocess_point_cloud(self, points: np.array, segImg: np.array) -> np.array:
        """
        Preprocess a point cloud by removing its points for the background, the points for the target and
        the points for the robot arm
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: np.array
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: np.array
        :return: the points of the point cloud with the points for the background, robot arm and target removed
        :rtype: np.array
        """
        # Points that have the same color as the first point in the point cloud are removed
        # Points that have the color [60, 180, 75] are removed, as this is the color used for the target point
        segImg = segImg.flatten()
        select_mask = segImg > 0
        points = points[select_mask]

        # p.addUserDebugPoints(points, np.tile([255, 0, 0], points.shape[0]).reshape(points.shape))
        # time.sleep(10)
        return points

    def _set_min_distances_of_robot_to_obstacle(self) -> None:
        """
        Compute and set the minimal euclidean distance between the coordinates of the centers of mass of the robot links
        to the obstacles points given by the scene point cloud.
        """
        # get coordinates of every links center of mass
        link_center_of_mass_coordinates = []
        for i in range(p.getNumJoints(self.RobotUid)):
            link_center_of_mass_coordinates.append(p.getLinkState(self.RobotUid, i)[0])
        self.link_positions = np.asarray(link_center_of_mass_coordinates, dtype=np.float32)
        # Compute minimal euclidean distance from the coordinates of every links center of mass to every obstacle point
        if self.points.shape[0] == 0:
            distances_to_obstacles = euclidean_distances(self.link_positions,
                                                         np.array([100, 100, 100])[na, :]).min(axis=1).round(10)
        else:
            distances_to_obstacles = euclidean_distances(self.link_positions, self.points).min(axis=1).round(10)

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)

    def _seg_mask_to_rgb(self, segImg: np.array) -> np.array:
        """
        Transform a pybullet segmentation mask into an RGB image by assigning a unqiue color to each class.
        :param segImg: a pybullet segmentation mask; its amount of entries should equal
        the product of the camera sensors pixel width and height
        :type segImg: np.array
        """
        segImg = segImg.flatten()
        return np.asarray(self.df_seg_to_rgb.loc[segImg])

    def _set_camera_matrices(self) -> None:
        """
        Set the camera sensors view and projection matrices by hardcoding the values within this method.
        """
        cameraEyePosition = np.array([0,
                                      0.75,
                                      0.75])

        # link_position = p.getLinkState(self.RobotUid, 4)[0]
        # cameraEyePosition = np.array([link_position[0] + p.readUserDebugParameter(self.x_offset),
        #                               link_position[1] + p.readUserDebugParameter(self.y_offset),
        #                               p.readUserDebugParameter(self.z_offset)])
        # The target of the camera sensor is the middle point of the work space
        cameraTargetPosition = np.array([(self.x_high_obs + self.x_low_obs) / 2,
                                         (self.y_high_obs + self.y_low_obs) / 2,
                                         (self.z_high_obs + self.z_low_obs) / 2])

        # The up-vector is chosen so that it is orthogonal to the vector between the camera eye and the camera target
        cameraUpVector = np.array([0, -0.6, 1])

        # Set the view matrix with hardcoded values
        self.camera_view_matrix = p.computeViewMatrix(
            cameraEyePosition, cameraTargetPosition, cameraUpVector
        )

        # # Add a debug line between the camera eye and the up vector in green and one between the eye and the target
        # # in red
        p.addUserDebugLine(cameraEyePosition, cameraTargetPosition, lineColorRGB=[1, 0, 0], lifeTime=0.2)
        p.addUserDebugLine(cameraEyePosition, cameraEyePosition + cameraUpVector, lineColorRGB=[0, 1, 0], lifeTime=0.2)
        # p.addUserDebugLine(cameraEyePosition, cameraEyePosition - np.array([0, 0, 2]), lineColorRGB=[0, 0, 1], lifeTime=0.2)
        # Set the projection matrix with hardcoded values
        self.camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=100,
            aspect=1,
            nearVal=0.1,
            farVal=1)

        # set transformation matrix to retrieve real world coordinates from pixel coordinates
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.camera_proj_matrix).reshape([4, 4], order='F'),
                                                      np.asarray(self.camera_view_matrix).reshape([4, 4], order='F')))

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

    def _add_obstacles(self):
        rand = np.float32(np.random.rand(3, ))
        target_x = self.x_low_obs + rand[0] * (self.x_high_obs - self.x_low_obs)
        target_y = self.y_low_obs + rand[1] * (self.y_high_obs - self.y_low_obs)
        target_z = self.z_low_obs + rand[2] * (self.z_high_obs - self.z_low_obs)
        target_position = [target_x, target_y, target_z]
        # print (target_position)
        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []
        for i in range(self.num_obstacles):
            obst_position = target_position
            val = False
            type = np.random.random()
            rate = np.random.random()
            if (rate < self.prob_obstacles or len(obsts) == 0) and (type > 0.5):
                cc = 0
                while not val:
                    cc += 1
                    rand = np.float32(np.random.rand(3, ))
                    obst_x = self.x_high_obs - rand[0] * (self.x_high_obs - self.x_low_obs)
                    obst_y = self.y_high_obs - rand[1] * (self.y_high_obs - self.y_low_obs)
                    obst_z = self.z_high_obs - rand[2] * (self.z_high_obs - self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position) - np.asarray(obst_position))
                    diff_2 = abs(np.asarray(self.init_home) - np.asarray(obst_position))
                    val = (diff > 0.05).all() and (np.linalg.norm(diff) < 0.5) and (diff_2 > 0.05).all() and (
                            np.linalg.norm(diff_2) < 0.5)
                    if cc > 100:
                        val = True
                if cc <= 100:
                    halfExtents = list(np.float32(np.random.uniform(0.8, 1.2) * np.array(self.obstacle_box_size)))
                    obst_orientation = [[0.707, 0, 0, 0.707], [0, 0.707, 0, 0.707], [0, 0, 0.707, 0.707]]
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box(halfExtents),
                        baseCollisionShapeIndex=self._create_collision_box(halfExtents),
                        basePosition=obst_position,
                        baseOrientation=choice(obst_orientation)
                    )
                    obsts.append(obst_id)
            if (rate < self.prob_obstacles or len(obsts) == 0) and (type <= 0.5):
                cc = 0
                while not val:
                    cc += 1
                    rand = np.float32(np.random.rand(3, ))
                    obst_x = self.x_high_obs - rand[0] * (self.x_high_obs - self.x_low_obs)
                    obst_y = self.y_high_obs - rand[1] * 0.5 * (self.y_high_obs - self.y_low_obs)
                    obst_z = self.z_high_obs - rand[2] * (self.z_high_obs - self.z_low_obs)
                    obst_position = [obst_x, obst_y, obst_z]
                    diff = abs(np.asarray(target_position) - np.asarray(obst_position))
                    diff_2 = abs(np.asarray(self.init_home) - np.asarray(obst_position))
                    val = (diff > 0.05).all() and (np.linalg.norm(diff) < 0.4) and (diff_2 > 0.05).all() and (
                            np.linalg.norm(diff_2) < 0.4)
                    if cc > 100:
                        val = True
                if cc <= 100:
                    radius = np.float32(np.random.uniform(0.8, 1.2)) * self.obstacle_sphere_radius
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_sphere(radius),
                        baseCollisionShapeIndex=self._create_collision_sphere(radius),
                        basePosition=obst_position,
                    )
                    obsts.append(obst_id)

                    assert len(obsts) != 0
        return target_position, obsts

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

        self.init_home, self.init_orn = self._set_home()

        self.target_position, self.obsts = self._add_obstacles()

        if self.extra_obst:
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

        # set point cloud image of obstacles
        # get depth image and segmentation mask
        depth, seg = self._get_image()

        # compute point cloud of obstacles
        self.points = self._depth_img_to_point_cloud(depth)

        # preprocess point cloud
        self.points = self._prepreprocess_point_cloud(self.points, seg)

        # reset
        self.step_counter = 0
        self.collided = False

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
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
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        # get position observation
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]

        self.current_joint_position = [0]

        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

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

        # do this step in pybullet
        p.stepSimulation()

        # if the simulation starts with the robot arm being inside an obstacle, reset again
        if p.getContactPoints(self.RobotUid) != ():
            self.reset()

        # input("Press ENTER")

        return self._get_obs()

    def step(self, action):
        # set a coefficient to prevent the action from being too large
        self.action = action
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        droll = action[3] * dv
        dpitch = action[4] * dv
        dyaw = action[5] * dv

        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        current_rpy = euler_from_quaternion(self.current_orn)
        new_robot_pos = [self.current_pos[0] + dx,
                         self.current_pos[1] + dy,
                         self.current_pos[2] + dz]
        new_robot_rpy = [current_rpy[0] + droll,
                         current_rpy[1] + dpitch,
                         current_rpy[2] + dyaw]
        self.motionexec.go_to_target(new_robot_pos, new_robot_rpy)

        if self.extra_obst:
            barr_pos = np.asarray(p.getBasePositionAndOrientation(self.barrier)[0])
            if self.moving_xy == 0:
                if barr_pos[0] > self.x_high_obs or barr_pos[0] < self.x_low_obs:
                    self.direction = -self.direction
                barr_pos[0] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])
            if self.moving_xy == 1:
                if barr_pos[1] > self.y_high_obs or barr_pos[1] < self.y_low_obs:
                    self.direction = -self.direction
                barr_pos[1] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])

        # update current pose
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        #
        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # # print (self.obs_rays)
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        # print (self.indicator)
        # check collision
        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])
            if len(contacts) > 0:
                self.collided = True

        p.stepSimulation()
        if self.is_good_view:
            time.sleep(0.04)

        self.step_counter += 1
        # input("Press ENTER")
        return self._reward()

    def _reward(self):
        reward = 0

        # distance between torch head and target postion
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos)) - np.asarray(self.target_position), ord=None)
        # print(self.distance)
        # check if out of boundary
        x = self.current_pos[0]
        y = self.current_pos[1]
        z = self.current_pos[2]
        out = bool(
            x < self.x_low_obs
            or x > self.x_high_obs
            or y < self.y_low_obs
            or y > self.y_high_obs
            or z < self.z_low_obs
            or z > self.z_high_obs
        )
        # check shaking
        shaking = 0
        if len(self.past_distance) >= 10:
            arrow = []
            for i in range(0, 9):
                arrow.append(0) if self.past_distance[i + 1] - self.past_distance[i] >= 0 else arrow.append(1)
            for j in range(0, 8):
                if arrow[j] != arrow[j + 1]:
                    shaking += 1
        reward -= shaking * 0.005

        # success
        is_success = False
        if out:
            self.terminated = True
            reward += -5
        elif self.collided:
            self.terminated = True
            reward += -10
        elif self.distance < self.distance_threshold:
            self.terminated = True
            is_success = True
            self.success_counter += 1
            reward += 10
        # not finish when reaches max steps
        elif self.step_counter >= self.max_steps_one_episode:
            print(self.distance)
            self.terminated = True
            reward += -1
        # this episode goes on
        else:
            self.terminated = False
            reward += -0.01 * self.distance

            # add reward for distance to obstacles
            reward += -1 * np.exp(-15 * self.distances_to_obstacles.min())

        info = {'step': self.step_counter,
                'out': out,
                'distance': self.distance,
                'reward': reward,
                'collided': self.collided,
                'shaking': shaking,
                'is_success': is_success,
                "min_distance_to_obstacles": self.distances_to_obstacles.min()}

        if self.terminated:
            print(info)
            # logger.debug(info)
        return self._get_obs(), reward, self.terminated, info

    def _get_obs(self):
        # set distsance to obstacles
        self._set_min_distances_of_robot_to_obstacle()

        # Set other observations
        self.state[0:6] = self.current_joint_position[1:]
        self.state[6:9] = np.asarray(self.target_position) - np.asarray(self.current_pos)
        self.state[9:13] = self.current_orn
        self.distance = np.linalg.norm(np.asarray(list(self.current_pos)) - np.asarray(self.target_position), ord=None)
        self.past_distance.append(self.distance)

        if len(self.past_distance) > 10:
            self.past_distance.popleft()
        self.state[13] = self.distance

        return {
            'position': self.state,
            # 'indicator': self.indicator,
            "distances_to_obstacles": self.distances_to_obstacles,
            "link_center_of_mass_coordinates": self.link_center_of_mass_coordinates
        }


class Env_V2(gym.Env):
    """
    Differences to Env_V1:
    - added distances to obstacles and distance to target to observations
    - added penalty for shaking
    - make it so there is always at least one obstacle
    - added additional points along the robot arm and head for the robot skeleton
    - generate worlds according to StaticImproveEnv
    - changed reward for distance to target to Huber Loss and added reward for motion size
    - some efficiency improvements
    """

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
            num_obstacles: int = 3,
            prob_obstacles: float = 0.8,
            obstacle_box_size: list = [0.04, 0.04, 0.002],
            obstacle_sphere_radius: float = 0.04
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
        self.action = None
        self.previous_action = np.zeros(6)
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)  # angular velocities

        # parameters for spatial infomation
        self.home = [0, np.pi / 2, -np.pi / 6, -2 * np.pi / 3, -4 * np.pi / 9, np.pi / 2, 0.0]
        self.target_position = None
        self.obsts = []
        self.current_pos = None
        self.current_orn = None
        self.current_joint_position = None
        self.vel_checker = 0
        self.past_distance = deque([])

        # observation space
        self.joint_positions = np.zeros(6, dtype=np.float32)
        # self.joint_angular_velocities = np.zeros(6, dtype=np.float32)
        self.end_effector_position = np.zeros(3, dtype=np.float32)
        self.target_position = np.zeros(3, dtype=np.float32)
        self.obstacle_points = np.zeros((10, 3), dtype=np.float32)
        self.robot_skeleton = np.zeros((10, 3), dtype=np.float32)
        self.end_effector_orn = np.zeros((4,), dtype=np.float32)
        self.distances_to_obstacles = np.zeros((10,), dtype=np.float32)
        self.distance = np.zeros((1,), dtype=np.float32)

        obs_spaces = {
            "joint_positions": spaces.Box(low=-3.15, high=3.15, shape=(6,)),
            # "joint_angular_velocities": spaces.Box(low=-1, high=1, shape=(6,)),
            "target_position": spaces.Box(low=-5, high=5, shape=(3,)),
            "obstacle_points": spaces.Box(low=-5, high=5, shape=(10, 3)),
            "robot_skeleton": spaces.Box(low=-5, high=5, shape=(10, 3)),  # this also contains the end effector position
            "end_effector_orn": spaces.Box(low=-5, high=5, shape=(4,)),
            "distances_to_obtacles": spaces.Box(low=-3, high=3, shape=(10,)),
            "distance_to_target": spaces.Box(low=-3, high=3, shape=(1,))
        }
        self.observation_space = spaces.Dict(obs_spaces)

        # set DataFrame for transforming segmentation mask to an RGB image
        translation_dic_red = pandas.DataFrame.from_dict(
            {-1: 230, 0: 60, 1: 255, 2: 0, 3: 245, 4: 145, 5: 70, 6: 240, 7: 210, 8: 250, 9: 0,
             10: 220, 11: 170}, orient="index", columns=["r"])
        translation_dic_green = pandas.DataFrame.from_dict(
            {-1: 25, 0: 180, 1: 225, 2: 130, 3: 130, 4: 30, 5: 240, 6: 50, 7: 245, 8: 190, 9: 128,
             10: 190, 11: 110}, orient="index", columns=["g"])
        translation_dic_blue = pandas.DataFrame.from_dict(
            {-1: 75, 0: 75, 1: 25, 2: 200, 3: 48, 4: 180, 5: 240, 6: 230, 7: 60, 8: 212, 9: 128,
             10: 255, 11: 40}, orient="index", columns=["b"])
        self.df_seg_to_rgb = pandas.concat([translation_dic_red, translation_dic_green, translation_dic_blue], axis=1)

        # step counter
        self.step_counter = 0
        # max steps in one episode
        self.max_steps_one_episode = max_steps_one_episode
        # whether collision
        self.collided = None
        # path to urdf of robot arm
        self.urdf_root_path = './ur5_description/urdf/ur5.urdf'
        # link indexes
        self.base_link = 1
        self.effector_link = 7
        # obstacles
        self.num_obstacles = num_obstacles
        self.prob_obstacles = prob_obstacles
        self.obstacle_box_size = obstacle_box_size
        self.obstacle_sphere_radius = obstacle_sphere_radius

        # # for debugging camera
        # self.x_offset = p.addUserDebugParameter("x", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("y", -2, 2, 0)
        # self.z_offset = p.addUserDebugParameter("z", -2, 2, 0)
        # set image width and height
        self._set_camera_matrices()
        self.img_width = 176
        self.img_height = 120

        # parameters of augmented targets for training
        if self.is_train:
            self.distance_threshold = 0.4
            self.distance_threshold_last = 0.4
            self.distance_threshold_increment_p = 0  # 0.0001
            self.distance_threshold_increment_m = 0.01
            self.distance_threshold_max = 0.4
            self.distance_threshold_min = 0.00001
        # parameters of augmented targets for testing
        else:
            self.distance_threshold = 0.02
            self.distance_threshold_last = 0.02
            self.distance_threshold_increment_p = 0.0
            self.distance_threshold_increment_m = 0.0
            self.distance_threshold_max = 0.02
            self.distance_threshold_min = 0.02

        self.episode_counter = 0
        self.episode_interval = 50
        self.success_counter = 0

    def _get_image(self) -> Tuple[List[float], List[int]]:
        """
        Retrieve a depth image and a segmentation mask from the camera sensor.
        :return: The depth image and the segmentation mask
        :rtype: Tuple[List[float], List[int]]
        """
        depthImg, segImg = p.getCameraImage(
            width=self.img_width,
            height=self.img_height,
            viewMatrix=self.camera_view_matrix,
            projectionMatrix=self.camera_proj_matrix)[3:]

        return depthImg, segImg

    def _depth_img_to_point_cloud(self, depth: np.array) -> np.array:
        """
        Compute a point cloud from a given depth image. The computation is done according to this stackoverflow post:
        https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        :param depth: input depth image;
        the amount of points in the image should equal the product of the camera sensors pixel width and height
        :type depth: np.array
        :return: The point cloud in the shape [width x height, 3]
        :rtype: np.array
        """
        # set width and height
        W = np.arange(0, self.img_width)
        H = np.arange(0, self.img_height)

        # compute pixel coordinates
        X = ((2 * W - self.img_width) / self.img_width)[na, :].repeat(self.img_height, axis=0).flatten()[:, na]
        Y = (-1 * (2 * H - self.img_height) / self.img_height)[:, na].repeat(self.img_width, axis=1).flatten()[:, na]
        Z = (2 * depth - 1).flatten()[:, na]

        # transform pixel coordinates into real world ones
        num_of_pixels = self.img_width * self.img_height
        PixPos = np.concatenate([X, Y, Z, np.ones(num_of_pixels)[:, na]], axis=1)
        points = np.tensordot(self.tran_pix_world, PixPos, axes=(1, 1)).swapaxes(0, 1)
        points = (points / points[:, 3][:, na])[:, 0:3]

        return points

    def _prepreprocess_point_cloud(self, points: np.array, segImg: np.array) -> np.array:
        """
        Preprocess a point cloud by removing its points for the background, the points for the target and
        the points for the robot arm
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: np.array
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: np.array
        :return: the points of the point cloud with the points for the background, robot arm and target removed
        :rtype: np.array
        """
        # Points that have the same color as the first point in the point cloud are removed
        # Points that have the color [60, 180, 75] are removed, as this is the color used for the target point
        segImg = segImg.flatten()
        select_mask = segImg > 0
        points = points[select_mask]

        # p.addUserDebugPoints(points, np.tile([255, 0, 0], points.shape[0]).reshape(points.shape))
        # time.sleep(100)
        return points

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
        # for point in self.robot_skeleton:
        #     p.addUserDebugLine(point, np.asarray(point) + np.array([0, 0, 2]), lineColorRGB=[255, 0, 0], lifeTime=5)
        # time.sleep(5)

    def _set_min_distances_of_robot_to_obstacle(self) -> None:
        """
        Compute the minimal distances from the robot skeleton to the obstacle points. Also determine the points
        that are closest to each point in the skeleton.
        """

        # Compute minimal euclidean distances from the robot skeleton to the obstacle points
        if self.points.shape[0] == 0:
            self.obstacle_points = np.repeat(np.array([0, 0, -2])[na, :], 10, axis=0)
            distances_to_obstacles = euclidean_distances(self.robot_skeleton,
                                                         np.array([0, 0, -2])[na, :]).min(axis=1).round(10)
        else:
            distances = euclidean_distances(self.robot_skeleton, self.points)
            self.obstacle_points = self.points[distances.argmin(axis=1)]
            distances_to_obstacles = distances.min(axis=1).round(10)

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)

    def _seg_mask_to_rgb(self, segImg: np.array) -> np.array:
        """
        Transform a pybullet segmentation mask into an RGB image by assigning a unqiue color to each class.
        :param segImg: a pybullet segmentation mask; its amount of entries should equal
        the product of the camera sensors pixel width and height
        :type segImg: np.array
        """
        segImg = segImg.flatten()
        return np.asarray(self.df_seg_to_rgb.loc[segImg])

    def _set_camera_matrices(self) -> None:
        """
        Set the camera sensors view and projection matrices by hardcoding the values within this method.
        """
        cameraEyePosition = np.array([0, 0.75, 0.75])

        # link_position = p.getLinkState(self.RobotUid, 4)[0]
        # cameraEyePosition = np.array([p.readUserDebugParameter(self.x_offset),
        #                               p.readUserDebugParameter(self.y_offset),
        #                               p.readUserDebugParameter(self.z_offset)])
        # The target of the camera sensor is the middle point of the work space
        cameraTargetPosition = np.array([(self.x_high_obs + self.x_low_obs) / 2,
                                         (self.y_high_obs + self.y_low_obs) / 2,
                                         (self.z_high_obs + self.z_low_obs) / 2])

        # The up-vector is chosen so that it is orthogonal to the vector between the camera eye and the camera target
        cameraUpVector = np.array([0, -0.6, 1])

        # Set the view matrix with hardcoded values
        self.camera_view_matrix = p.computeViewMatrix(
            cameraEyePosition, cameraTargetPosition, cameraUpVector
        )

        # # Add a debug line between the camera eye and the up vector in green and one between the eye and the target
        # # in red
        p.addUserDebugLine(cameraEyePosition, cameraTargetPosition, lineColorRGB=[1, 0, 0], lifeTime=0.2)
        p.addUserDebugLine(cameraEyePosition, cameraEyePosition + cameraUpVector, lineColorRGB=[0, 1, 0], lifeTime=0.2)
        # p.addUserDebugLine(cameraEyePosition, cameraEyePosition - np.array([0, 0, 2]), lineColorRGB=[0, 0, 1], lifeTime=0.2)
        # Set the projection matrix with hardcoded values
        self.camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=100,
            aspect=1,
            nearVal=0.1,
            farVal=1)

        # set transformation matrix to retrieve real world coordinates from pixel coordinates
        self.tran_pix_world = np.linalg.inv(np.matmul(np.asarray(self.camera_proj_matrix).reshape([4, 4], order='F'),
                                                      np.asarray(self.camera_view_matrix).reshape([4, 4], order='F')))

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
            if np.random.random() > 1 - self.prob_obstacles:
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
                if i == 1:
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box([0.001, 0.08, 0.06]),
                        baseCollisionShapeIndex=self._create_collision_box([0.001, 0.05, 0.05]),
                        basePosition=position
                    )
                    obsts.append(obst_id)
                if i == 2:
                    obst_id = p.createMultiBody(
                        baseMass=0,
                        baseVisualShapeIndex=self._create_visual_box([0.08, 0.001, 0.06]),
                        baseCollisionShapeIndex=self._create_collision_box([0.05, 0.001, 0.05]),
                        basePosition=position
                    )
                    obsts.append(obst_id)

        # so we always have at least one obstacle
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
            if i == 1:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_box([0.001, 0.08, 0.06]),
                    baseCollisionShapeIndex=self._create_collision_box([0.001, 0.05, 0.05]),
                    basePosition=position
                )
                obsts.append(obst_id)
            if i == 2:
                obst_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=self._create_visual_box([0.08, 0.001, 0.06]),
                    baseCollisionShapeIndex=self._create_collision_box([0.05, 0.001, 0.05]),
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

        self.init_home, self.init_orn = self._set_home()
        self.target_position = np.asarray(self._set_target_position(), dtype=np.float32)
        self.obsts = self._add_obstacles()

        # for adding moving obstacle
        if self.extra_obst:
            self.direction = choice([-1, 1])
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

        # set point cloud image of obstacles
        # get depth image and segmentation mask
        depth, seg = self._get_image()

        # compute point cloud of obstacles
        self.points = self._depth_img_to_point_cloud(depth)

        # preprocess point cloud
        self.points = self._prepreprocess_point_cloud(self.points, seg)

        # reset
        self.step_counter = 0
        self.collided = False

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
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
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        # get position observation
        self.current_pos = np.asarray(p.getLinkState(self.RobotUid, self.effector_link)[4], dtype=np.float32)
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]

        # for lidar
        # self.wrist3_pos = p.getLinkState(self.RobotUid,6)[4]
        # self.wrist3_orn = p.getLinkState(self.RobotUid,6)[5]
        # self.wrist2_pos = p.getLinkState(self.RobotUid,5)[4]
        # self.wrist2_orn = p.getLinkState(self.RobotUid,5)[5]
        # self.wrist1_pos = p.getLinkState(self.RobotUid,4)[4]
        # self.wrist1_orn = p.getLinkState(self.RobotUid,4)[5]
        # self.arm3_pos = p.getLinkState(self.RobotUid,3)[4]
        # self.arm3_orn = p.getLinkState(self.RobotUid,3)[5]

        self.current_joint_position = [0]
        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        # print (self.indicator)

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.6 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.6 and self.distance_threshold > self.distance_threshold_min:
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

        # counter for number of timesteps spent in distance threshold
        self.n_timesteps_in_threshold = 0

        return self._get_obs()

    def step(self, action):
        self.action = action

        # old implementation where the action is interpreted as a position and rotation change
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        droll = action[3] * dv
        dpitch = action[4] * dv
        dyaw = action[5] * dv

        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]
        current_rpy = euler_from_quaternion(self.current_orn)
        new_robot_pos = [self.current_pos[0] + dx,
                         self.current_pos[1] + dy,
                         self.current_pos[2] + dz]
        new_robot_rpy = [current_rpy[0] + droll,
                         current_rpy[1] + dpitch,
                         current_rpy[2] + dyaw]
        self.motionexec.go_to_target(new_robot_pos, new_robot_rpy)

        # for moving obstacle
        if self.extra_obst:
            barr_pos = np.asarray(p.getBasePositionAndOrientation(self.barrier)[0])
            if self.moving_xy == 0:
                if barr_pos[0] > self.x_high_obs or barr_pos[0] < self.x_low_obs:
                    self.direction = -self.direction
                barr_pos[0] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])
            if self.moving_xy == 1:
                if barr_pos[1] > self.y_high_obs or barr_pos[1] < self.y_low_obs:
                    self.direction = -self.direction
                barr_pos[1] += self.direction * self.moving_obstacle_speed * dv
                p.resetBasePositionAndOrientation(self.barrier, barr_pos,
                                                  p.getBasePositionAndOrientation(self.barrier)[1])

        # update current pose
        self.current_pos = p.getLinkState(self.RobotUid, self.effector_link)[4]
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]

        # for lidar
        # self.wrist3_pos = p.getLinkState(self.RobotUid,6)[4]
        # self.wrist3_orn = p.getLinkState(self.RobotUid,6)[5]
        # self.wrist2_pos = p.getLinkState(self.RobotUid,5)[4]
        # self.wrist2_orn = p.getLinkState(self.RobotUid,5)[5]
        # self.wrist1_pos = p.getLinkState(self.RobotUid,4)[4]
        # self.wrist1_orn = p.getLinkState(self.RobotUid,4)[5]
        # self.arm3_pos = p.getLinkState(self.RobotUid,3)[4]
        # self.arm3_orn = p.getLinkState(self.RobotUid,3)[5]

        self.current_joint_position = [0]
        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # # print (self.obs_rays)
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        # print (self.indicator)
        # check collision

        for i in range(len(self.obsts)):
            contacts = p.getContactPoints(bodyA=self.RobotUid, bodyB=self.obsts[i])
            if len(contacts) > 0:
                self.collided = True

        p.stepSimulation()

        if self.is_good_view:
            time.sleep(0.02)

        self.step_counter += 1
        # input("Press ENTER")
        return self._reward()

    def _reward(self):
        reward = 0

        # set parameters
        lambda_1 = 10
        lambda_2 = 1
        lambda_3 = 0.008
        lambda_4 = 0.01
        dirac = 0.35
        k = 8
        d_ref = 0.28

        # set observations
        self._set_obs()

        # set motion change
        a = self.action - self.previous_action
        self.previous_action = self.action
        # calculating Huber loss for distance of end effector to target
        if self.distance < dirac:
            R_E_T = 1 / 2 * (self.distance[0] ** 2)
        else:
            R_E_T = dirac * (self.distance[0] - 1 / 2 * dirac)
        R_E_T = -R_E_T

        # calculating the distance between robot and obstacle
        R_R_O = -(d_ref / (self.distances_to_obstacles.min() + d_ref)) ** k

        # calculate motion size
        R_A = - np.sum(np.square(a))

        # check shaking
        shaking = 0
        if len(self.past_distance) >= 10:
            arrow = []
            for i in range(0, 9):
                arrow.append(0) if self.past_distance[i + 1] - self.past_distance[i] >= 0 else arrow.append(1)
            for j in range(0, 8):
                if arrow[j] != arrow[j + 1]:
                    shaking += 1

        # calculate reward
        reward += lambda_1 * R_E_T + lambda_2 * R_R_O + lambda_3 * R_A - lambda_4 * shaking

        # check if out of boundery
        x = self.end_effector_position[0]
        y = self.end_effector_position[1]
        z = self.end_effector_position[2]
        d = 0.05
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
            reward += 50
        elif self.collided:
            self.terminated = True
            reward += -50
        elif self.step_counter >= self.max_steps_one_episode:
            reward += -10
            self.terminated = True
        elif out:
            reward -= 25
            # self.terminated = True

        info = {'step': self.step_counter,
                "n_episode": self.episode_counter,
                "success_counter": self.success_counter,
                'distance': self.distance[0],
                "min_distance_to_obstacles": self.distances_to_obstacles.min(),
                "reward_1": lambda_1 * R_E_T,
                "reward_2": lambda_2 * R_R_O,
                "reward_3": lambda_3 * R_A,
                "reward_4": - lambda_4 * shaking,
                'reward': reward,
                'collided': self.collided,
                'is_success': is_success,
                'out': out
                }

        if self.terminated:
            print(info)
            # logger.debug(info)
        return self._get_obs(), reward, self.terminated, info

    def _set_obs(self):
        # set observations
        joint_positions = []
        joint_angular_velocities = []
        for i in range(p.getNumJoints(self.RobotUid)):
            joint_positions.append(p.getJointState(self.RobotUid, i)[0])
            joint_angular_velocities.append(p.getJointState(self.RobotUid, i)[1])
        self.joint_positions = np.asarray(joint_positions, dtype=np.float32)
        self.joint_angular_velocities = np.asarray(joint_angular_velocities, dtype=np.float32)
        # remove the fixed joints
        self.joint_positions = self.joint_positions[1:7]
        self.joint_angular_velocities = self.joint_angular_velocities[1:7]
        # set end effector position
        self.end_effector_position = np.asarray(p.getLinkState(self.RobotUid, 7)[0], dtype=np.float32)
        # distance between torch head and target postion
        self.distance = np.linalg.norm(self.end_effector_position - self.target_position, ord=None)[na]

        #  set distances to obstacles and robot skeleton
        self._set_robot_skeleton()
        self._set_min_distances_of_robot_to_obstacle()

        # set end effector orientation
        self.end_effector_orn = np.asarray(p.getLinkState(self.RobotUid, 7)[1], dtype=np.float32)
        # for shaking
        self.past_distance.append(self.distance)
        if len(self.past_distance) > 10:
            self.past_distance.popleft()

        # p.removeAllUserDebugItems()
        # for i in range(10):
        #     p.addUserDebugLine(self.obstacle_points[i, :], self.obstacle_points[i, :] + np.array([0, 0, 2]))

    def _get_obs(self):
        return {
            "joint_positions": self.joint_positions,
            #  "joint_angular_velocities": self.joint_angular_velocities,
            "target_position": self.target_position,
            "obstacle_points": self.obstacle_points,
            "robot_skeleton": self.robot_skeleton,
            "end_effector_orn": self.end_effector_orn,
            "distances_to_obtacles": self.distances_to_obstacles,
            "distance_to_target": self.distance
        }


class Env_V3(Env_V2):
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
            obstacle_box_size: list = [0.04, 0.04, 0.002],
            obstacle_sphere_radius: float = 0.06,
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

        # # for debugging camera
        # self.x_offset = p.addUserDebugParameter("x", -2, 2, 0)
        # self.y_offset = p.addUserDebugParameter("y", -2, 2, 0)
        # self.z_offset = p.addUserDebugParameter("z", -2, 2, 0)
        # set image width and height

        # set camera and image resolution
        self._set_camera_matrices()
        self.img_width = 176
        self.img_height = 120

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

    def _set_min_distances_of_robot_to_obstacle(self) -> None:
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
        distances = euclidean_distances(self.robot_skeleton, [self.obstacle_position]) - self.obstacle_radius
        #self.obstacle_points = self.points[distances.argmin(axis=1)]
        distances_to_obstacles = abs(distances.min(axis=1).round(10))

        self.distances_to_obstacles = distances_to_obstacles.astype(np.float32)

        # p.removeAllUserDebugItems()
        # p.addUserDebugPoints(self.obstacle_points, np.tile([255, 0, 0], self.obstacle_points.shape[0]).reshape(self.obstacle_points.shape),
        #                      pointSize=3)
        # time.sleep(20)

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

        # # transform action
        # joint_delta = action * self.joint_speed
        #
        # # add action to current state
        # joints_desired = self.joint_positions + joint_delta
        #
        # # check if desired joints would violate any joint range constraints
        # upper_limit_mask = joints_desired > self.joints_upper_limits
        # lower_limit_mask = joints_desired < self.joints_lower_limits
        # # set those joints to their respective max/min
        # joints_desired[upper_limit_mask] = self.joints_upper_limits[upper_limit_mask]
        # joints_desired[lower_limit_mask] = self.joints_lower_limits[lower_limit_mask]
        #
        # # execute movement by setting the desired joint state
        # self._movej(joints_desired)

        # for moving obstacle

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
        
        
        #R_E_T = -self.distance[0]

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
        self._set_min_distances_of_robot_to_obstacle()

        obs_id = self.obsts[0]

        obs_pos = p.getBasePositionAndOrientation(obs_id)[0]
        obs_vel = p.getBaseVelocity(obs_id)[0]
        self.obstacle_position = np.array(obs_pos, dtype=np.float32) #Only use position not orientation
        self.obstacle_velocity = np.array(obs_vel, dtype=np.float32)

    def _get_obs(self):
        return {
            "joint_positions": self.joint_positions,
            "joint_angular_velocities": self.joint_angular_velocities,
            "target_position": self.target_position,
            "end_effector_position": self.end_effector_position,
            "obstacle_position": self.obstacle_position,
            "obstacle_velocity": self.obstacle_velocity
        }

    def reset_test(self):
        self.t1 = time.time()
        p.resetSimulation()
        # planeId = p.loadURDF("plane.urdf")
        # print(time.time())
        self.recoder = []
        one_info = []
        if self.test_mode == 1:
            self.init_home, self.init_orn, self.target_position, self.obsts = self._test_1()
        elif self.test_mode == 2:
            self.init_home, self.init_orn, self.target_position, self.obsts = self._test_2()
        elif self.test_mode == 3:
            self.init_home = [0.25, 0.4, 0.3]
            self.init_orn = [np.pi, 0, np.pi]
            self.target_position = [0, 0.4, 0.25]
            target = p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
                basePosition=self.target_position,
            )
            self.obsts = self._test_3()
        # for adding moving obstacle
        if self.extra_obst:
            self.direction = choice([-1, 1])
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

        # set point cloud image of obstacles
        # get depth image and segmentation mask
        depth, seg = self._get_image()

        # compute point cloud of obstacles
        self.points = self._depth_img_to_point_cloud(depth)

        # preprocess point cloud
        self.points = self._prepreprocess_point_cloud(self.points, seg)

        # reset
        self.step_counter = 0
        self.collided = False
        self.ep_reward = 0
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
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
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        # get position observation
        self.current_pos = np.asarray(p.getLinkState(self.RobotUid, self.effector_link)[4], dtype=np.float32)
        self.current_orn = p.getLinkState(self.RobotUid, self.effector_link)[5]

        # for lidar
        # self.wrist3_pos = p.getLinkState(self.RobotUid,6)[4]
        # self.wrist3_orn = p.getLinkState(self.RobotUid,6)[5]
        # self.wrist2_pos = p.getLinkState(self.RobotUid,5)[4]
        # self.wrist2_orn = p.getLinkState(self.RobotUid,5)[5]
        # self.wrist1_pos = p.getLinkState(self.RobotUid,4)[4]
        # self.wrist1_orn = p.getLinkState(self.RobotUid,4)[5]
        # self.arm3_pos = p.getLinkState(self.RobotUid,3)[4]
        # self.arm3_orn = p.getLinkState(self.RobotUid,3)[5]

        self.current_joint_position = [0]
        # # get lidar observation
        # lidar_results = self._set_lidar_cylinder()
        # for i, ray in enumerate(lidar_results):
        #     self.obs_rays[i] = ray[2]
        # rc = RaysCauculator(self.obs_rays)
        # self.indicator = rc.get_indicator()

        # print (self.indicator)

        for i in range(self.base_link, self.effector_link):
            self.current_joint_position.append(p.getJointState(bodyUniqueId=self.RobotUid, jointIndex=i)[0])

        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.6 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.6 and self.distance_threshold > self.distance_threshold_min:
                self.distance_threshold -= self.distance_threshold_increment_m
            elif success_rate == 1 and self.distance_threshold == self.distance_threshold_min:
                self.distance_threshold == self.distance_threshold_min
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

    def _test_1(self):
        '''
        a simple barrier between init position and target
        '''

        init_home = [0.15, 0.4, 0.3]
        init_orn = [np.pi, 0, np.pi]
        target_position = [-0.15, 0.4, 0.3]
        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []
        obst_1 = p.createMultiBody(
            baseMass=0,
            # baseVisualShapeIndex=self._create_visual_box([0.002,0.1,0.05]),
            baseCollisionShapeIndex=self._create_collision_box([0.002, 0.1, 0.05]),
            basePosition=[0.0, 0.4, 0.3],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        obsts.append(obst_1)
        return init_home, init_orn, target_position, obsts

    def _test_2(self):
        init_home = [0.1, 0.3, 0.33]
        init_orn = [np.pi, 0, np.pi]
        target_position = [-0.3, 0.5, 0.25]
        target = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.02, rgbaColor=[1, 0, 0, 1]),
            basePosition=target_position,
        )
        obsts = []

        return init_home, init_orn, target_position, obsts

    def _test_3(self):
        obsts = []
        obst_1 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._create_visual_box([0.002, 0.1, 0.05]),
            baseCollisionShapeIndex=self._create_collision_box([0.002, 0.1, 0.05]),
            basePosition=[-0.1, 0.4, 0.26],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        obsts.append(obst_1)
        obst_2 = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self._create_visual_box([0.002, 0.1, 0.05]),
            baseCollisionShapeIndex=self._create_collision_box([0.002, 0.1, 0.05]),
            basePosition=[0.1, 0.4, 0.26],
            baseOrientation=choice([0.707, 0, 0, 0.707])
        )
        obsts.append(obst_2)
        return obsts

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


class Env_V4(Env_V3):
    "Trying to optimize the hyperparameters of Env_V3"

    def __init__(self, is_render: bool = False, is_good_view: bool = False, is_train: bool = True,
                 show_boundary: bool = True, add_moving_obstacle: bool = False, moving_obstacle_speed: float = 0.15,
                 moving_init_direction: int = -1, moving_init_axis: int = 0,
                 workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], max_steps_one_episode: int = 1024,
                 num_obstacles: int = 3, prob_obstacles: float = 0.8, obstacle_box_size: list = [0.04, 0.04, 0.002],
                 obstacle_sphere_radius: float = 0.04, test_mode: int = 0):
        super().__init__(is_render, is_good_view, is_train, show_boundary, add_moving_obstacle, moving_obstacle_speed,
                         moving_init_direction, moving_init_axis, workspace, max_steps_one_episode, num_obstacles,
                         prob_obstacles, obstacle_box_size, obstacle_sphere_radius, test_mode)

    def reset(self):
        p.resetSimulation()

        self.action = None
        self.init_home, self.init_orn = self._set_home()
        self.target_position = self._set_target_position()
        self.obsts = self._add_obstacles()
        if len(self.obsts) == 0:
            self.reset()
        self.target_position = np.asarray(self.target_position, dtype=np.float32)

        # for adding moving obstacle
        if self.extra_obst:
            self.direction = choice([-1, 1])
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

        # retrieve depth image and segmentation mask
        depth, seg = self._get_image()
        # compute point cloud of obstacles
        self.points = self._depth_img_to_point_cloud(depth)
        # preprocess point cloud
        self.points = self._prepreprocess_point_cloud(self.points, seg)
        if self.points.shape[0] == 0:
            self.reset()

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
        # robot goes to the initial position
        self.motionexec.go_to_target(self.init_home, self.init_orn)

        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.6 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.6 and self.distance_threshold > self.distance_threshold_min:
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

    def _reward(self):
        reward = 0

        # set parameters
        lambda_1 = 1
        lambda_2 = 15
        lambda_3 = 0.06
        k = 8
        d_ref = 0.33

        # set observations
        self._set_obs()
        # print(self.joint_angular_velocities)
        # reward for distance to target
        R_E_T = -self.distance[0]

        # reward for distance to obstacle
        R_R_O = -(d_ref / (self.distances_to_obstacles.min() + d_ref)) ** k

        # calculate motion size
        R_A = - np.sum(np.square(self.action))

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
            reward += 500
        elif self.collided:
            self.terminated = True
            reward += -500
        elif self.step_counter >= self.max_steps_one_episode:
            reward += -100
            self.terminated = True
        elif out:
            self.terminated = True
            reward -= 500

        self.ep_reward += reward

        info = {'step': self.step_counter,
                "n_episode": self.episode_counter,
                "success_counter": self.success_counter,
                'distance': self.distance[0],
                "min_distance_to_obstacles": self.distances_to_obstacles.min(),
                "reward_1": -lambda_1 * self.distance[0],
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


class Env_V5(Env_V3):

    def __init__(self, is_render: bool = False, is_good_view: bool = False, is_train: bool = True,
                 show_boundary: bool = True, add_moving_obstacle: bool = False, moving_obstacle_speed: float = 0.15,
                 moving_init_direction: int = -1, moving_init_axis: int = 0,
                 workspace: list = [-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], max_steps_one_episode: int = 1024,
                 num_obstacles: int = 3, prob_obstacles: float = 0.8, obstacle_box_size: list = [0.04, 0.04, 0.002],
                 obstacle_sphere_radius: float = 0.04, test_mode: int = 0):
        super().__init__(is_render, is_good_view, is_train, show_boundary, add_moving_obstacle, moving_obstacle_speed,
                         moving_init_direction, moving_init_axis, workspace, max_steps_one_episode, num_obstacles,
                         prob_obstacles, obstacle_box_size, obstacle_sphere_radius, test_mode)
        self.joint_1 = p.addUserDebugParameter("shoulder left right", -3, 3, 0)
        self.joint_2 = p.addUserDebugParameter("shoulder top bottom", -3, 3, 0)
        self.joint_3 = p.addUserDebugParameter("elbow", -3, 3, 0)
        self.joint_4 = p.addUserDebugParameter("4", -3, 3, 0)
        self.joint_5 = p.addUserDebugParameter("5", -3, 3, 0)
        self.joint_6 = p.addUserDebugParameter("6", -3, 3, 0)
        self.joints_home = []

    def _prepreprocess_point_cloud(self, points: np.array, segImg: np.array) -> np.array:
        """
        Preprocess a point cloud by removing its points for the background, the points for the target and
        the points for the robot arm
        :param points: an array containing the x, y and z coordinates
        of the point cloud in the shape [width x height, 3]
        :type points: np.array
        :param segImg: an array containing the segmentation mask given by pybullet; number of entries needs to equal
        width x height
        :type segImg: np.array
        :return: the points of the point cloud with the points for the background, robot arm and target removed
        :rtype: np.array
        """
        # Points that have the same color as the first point in the point cloud are removed
        # Points that have the color [60, 180, 75] are removed, as this is the color used for the target point
        segImg = segImg.flatten()
        select_mask = segImg > 1
        points = points[select_mask]

        # p.removeAllUserDebugItems()
        # p.addUserDebugPoints(points, np.tile([255, 0, 0], points.shape[0]).reshape(points.shape))
        # time.sleep(10)
        return points

    def set_joints(self):
        joint_pos = [p.readUserDebugParameter(self.joint_1),
                     p.readUserDebugParameter(self.joint_2),
                     p.readUserDebugParameter(self.joint_3),
                     p.readUserDebugParameter(self.joint_4),
                     p.readUserDebugParameter(self.joint_5),
                     p.readUserDebugParameter(self.joint_6)]
        for i in range(1, 7):
            p.resetJointState(self.RobotUid, i, joint_pos[i - 1])

    def _set_joint_home(self):
        joints_home = np.asarray([
            np.random.uniform(-1.7, -1.5),
            np.random.uniform(-1, -0.95),
            np.random.uniform(1.4, 1.6),
            np.random.uniform(-2.4, -1.8),
            np.random.uniform(-1.9, -1.6),
            0
        ])

        return joints_home

    def reset(self):
        p.resetSimulation()
        self.action = None

        # get initial joint configuration
        self.joints_home = self._set_joint_home()

        # for adding moving obstacle
        if self.extra_obst:
            self.direction = choice([-1, 1])
            self.moving_xy = choice([0, 1])
            self.barrier = self._add_moving_plate()
            self.obsts.append(self.barrier)

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
        self.RobotUid = p.loadURDF(self.urdf_root_path, basePosition=[0.0, self.y_low_obs - 0.4, self.z_low_obs],
                                   baseOrientation=baseorn,
                                   useFixedBase=True)

        # set joints to initial configuration
        for i in range(1, 7):
            p.resetJointState(self.RobotUid, i, self.joints_home[i - 1])

        # get init home
        self.init_home = p.getLinkState(self.RobotUid, 7)[0]

        # get target position and obstacles (they rely on init_home)
        self.target_position = self._set_target_position()
        self.obsts = self._add_obstacles()
        if len(self.obsts) == 0:
            self.reset()
        self.target_position = np.asarray(self.target_position, dtype=np.float32)

        # move robot out of the way before taking the image
        for i in range(1, 7):
            p.resetJointState(self.RobotUid, i, 0)

        # retrieve depth image and segmentation mask
        depth, seg = self._get_image()
        # compute point cloud of obstacles
        self.points = self._depth_img_to_point_cloud(depth)
        # preprocess point cloud
        self.points = self._prepreprocess_point_cloud(self.points, seg)
        if self.points.shape[0] == 0:
            self.reset()

        # set init home configurations again
        for i in range(1, 7):
            p.resetJointState(self.RobotUid, i, self.joints_home[i - 1])

        # update soft goal
        self.episode_counter += 1
        if self.episode_counter % self.episode_interval == 0:
            self.distance_threshold_last = self.distance_threshold
            success_rate = self.success_counter / self.episode_interval
            self.success_counter = 0
            if success_rate < 0.6 and self.distance_threshold < self.distance_threshold_max:
                self.distance_threshold += self.distance_threshold_increment_p
            elif success_rate >= 0.6 and self.distance_threshold > self.distance_threshold_min:
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

    def _reward(self):
        reward = 0

        # set parameters
        lambda_1 = 1
        lambda_2 = 18
        lambda_3 = 0.06
        k = 8
        d_ref = 0.33

        # set observations
        self._set_obs()
        # print(self.joint_angular_velocities)
        # reward for distance to target
        R_E_T = -self.distance[0]

        # reward for distance to obstacle
        R_R_O = -(d_ref / (self.distances_to_obstacles.min() + d_ref)) ** k

        # calculate motion size
        R_A = - np.sum(np.square(self.action))

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
            reward += 500
        elif self.collided:
            self.terminated = True
            reward += -500
        elif self.step_counter >= self.max_steps_one_episode:
            reward += -100
            self.terminated = True
        elif out:
            self.terminated = True
            reward -= 500

        self.ep_reward += reward

        info = {'step': self.step_counter,
                "n_episode": self.episode_counter,
                "success_counter": self.success_counter,
                'distance': self.distance[0],
                "min_distance_to_obstacles": self.distances_to_obstacles.min(),
                "reward_1": -lambda_1 * self.distance[0],
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


if __name__ == '__main__':
    env = Env_V2(is_render=True, is_good_view=False, add_moving_obstacle=True)
    episodes = 500
    for episode in range(episodes):
        state = env.reset()
        done = False
        i = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            # print(info)
