import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
import cv2
from calib import Calibration

class Projection():
    def __init__(self) -> None:
        self.calib_path = "/home/rohiitb/semantic_mapping_icp/dataset/calibration"
        self.img_dataset_path = "/home/rohiitb/semantic_mapping_icp/dataset/camera/images"
        self.pcd_dataset_path = "/home/rohiitb/semantic_mapping_icp/dataset/lidar/pcd_points"

        self.calib = Calibration(self.calib_path)

    def get_single_data(self, idx):
        pcd_data = os.listdir(self.pcd_dataset_path)
        pcd_data.sort()
        img_data = os.listdir(self.img_dataset_path)
        img_data.sort()

        pcd_load = o3d.io.read_point_cloud(f"{self.pcd_dataset_path}/{pcd_data[idx]}")
        # o3d.visualization.draw_geometries([pcd_load])
        # raise
        img = cv2.imread(f"{self.img_dataset_path}/{img_data[idx]}")

        points = np.asarray(pcd_load.points)
        points = np.hstack((points, np.ones(len(points)).reshape(-1,1)))
        
        points_transformed = (np.linalg.inv(self.calib.T_cam_2_lidar) @ points.T).T
        
        points_transformed = points_transformed[points_transformed[:,2]>0]
        
        pts_cam_frame = self.calib.P_rect_camera_0 @ self.calib.R_rect_camera_0 @ points_transformed.T
        pts_cam_frame = pts_cam_frame[:3] / pts_cam_frame[2]

        plt.scatter(pts_cam_frame[0], pts_cam_frame[1],s=1,marker='o')
        plt.imshow(img)
        plt.show()
        # print("Here done: ", pts_cam_frame)

        

    
        


    def proj_lid_2_image(pcd, image):
        pass
        

if __name__ == "__main__":
    proj = Projection()
    proj.get_single_data(100)