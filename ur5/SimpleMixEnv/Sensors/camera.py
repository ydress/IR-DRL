import numpy as np

class Camera():
    
    def __init__(self, 
                 pybullet_env):
        self.pybullet_env = pybullet_env
    
    def _get_camera_data(self, cameraType="RGB"):
        width = 480
        height = 480
        view_matrix = self.pybullet_env.computeViewMatrix(
                        cameraEyePosition=[0, 0, 3],
                        cameraTargetPosition=[0, 0, 0],
                        cameraUpVector=[0, 1, 0])
        projection_matrix = self.pybullet_env.computeProjectionMatrixFOV(
                        fov=45.0,
                        aspect=1.0,
                        nearVal=0.1,
                        farVal=3.1)
        images = self.pybullet_env.getCameraImage(width = width,
                        height = height,
                        viewMatrix=view_matrix,
                        projectionMatrix=projection_matrix)
        #depth_buffer_tiny = np.reshape(images[3], [width, height])
        #depth_tiny = far * near / (far - (far - near) * depth_buffer_tiny)
        rgb_tiny = np.reshape(images[2], (height, width, 4)) * 1. / 255.
        
        if cameraType == "RGB":
            return rgb_tiny
        elif cameraType == "Depth":
            return
            #return depth_tiny
        else:
            raise Exception("Unknown camera type")