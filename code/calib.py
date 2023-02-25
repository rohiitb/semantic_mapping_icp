import numpy as np

class Calibration():
    def __init__(self, path) -> None:
        self.path = path
        self.path_calib_cam = self.path + "/perspective.txt"
        self.path_calib_cam2lid = self.path + "/calib_cam_to_velo.txt"
        self.data_dict = self.extract_calib()
        
        K_00 = self.get_K_camera_0()
        self.K_00 = K_00

        P_rect_camera_0 = self.get_P_rect_camera_0()
        self.P_rect_camera_0 = P_rect_camera_0

        R_rect_camera_0 = self.get_R_rect_camera_0()
        self.R_rect_camera_0 = R_rect_camera_0

        T_cam_2_lidar = self.get_T_cam_2_lidar()
        self.T_cam_2_lidar = T_cam_2_lidar

        self.image_size = (1408, 376)

    def get_T_cam_2_lidar(self):
        with open(self.path_calib_cam2lid, 'r') as f:
            content = f.read().splitlines()[0]
            content = content.split(' ')
        vals = np.array([float(val) for val in content]).reshape(3,4)
        return vals
        
    def extract_calib(self):
        data = {}
        with open(self.path_calib_cam, 'r') as f:
            content = f.read().splitlines()
            for i in range(len(content)):
                key, value = content[i].split(':')
                key = key.split(' ')[0]
                value = [float(val) for val in value.split(' ') if val != '']
                data[key] = value    
        return data        

    def get_K_camera_0(self):
        K = np.array(self.data_dict['K_00']).reshape(3,3)
        return K
        
    def get_P_rect_camera_0(self):
        P = np.array(self.data_dict['P_rect_00']).reshape(3,4)
        return P

    def get_R_rect_camera_0(self):
        R = np.array(self.data_dict['R_rect_00']).reshape(3,3)
        return R


if __name__ == "__main__":
    path = "/home/rohiitb/semantic_mapping_icp/dataset/calibration"
    calib = Calibration(path)
