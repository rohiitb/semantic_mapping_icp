import numpy as np
import open3d as o3d
import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import cv2
from utils import *
from calib import Calibration


class BEV():
    def __init__(self) -> None:
        self.dir_path = dirname(dirname(abspath(__file__)))
        self.calib_path = f"{self.dir_path}/dataset/calibration"
        self.img_dataset_path = f"{self.dir_path}/dataset/camera/images"
        self.pcd_dataset_path = f"{self.dir_path}/dataset/lidar/pcd_points"
        # self.painted_pcd_rgb = f"{self.dir_path}/dataset/painted_points_rgb"
        self.segmented_pcd = f"{self.dir_path}/dataset/segmented_pcd"

        self.segmented_img_dataset_path = f"{self.dir_path}/dataset/segmented_imgs"
        self.calib = Calibration(self.calib_path)
        
        self.voxel_size = 0.05
        
        all_pcds = self.load_pcds()
        self.pcds_down = all_pcds

    def get_single_data(self, idx):
        pcd_data = os.listdir(self.pcd_dataset_path)
        pcd_data.sort()
        img_data = os.listdir(self.segmented_img_dataset_path)
        img_data.sort()

        pcd_load = o3d.io.read_point_cloud(f"{self.pcd_dataset_path}/{pcd_data[idx]}")
        points = np.asarray(pcd_load.points)
        img = cv2.imread(f"{self.segmented_img_dataset_path}/{img_data[idx]}")
        return points, img

    def paint_pointcloud(self, idx, visualize=False):
        
        points, img = self.get_single_data(idx)
        points = make_pts_homogenous(points)
        points_transformed, front_idx = lidar_2_cam(points, self.calib.T_cam_2_lidar)
        pts_cam_frame, outside_idx = proj_lidar_2_img(points_transformed, self.calib.P_rect_camera_0, self.calib.R_rect_camera_0, self.calib.image_size)
        rgb = get_rgb(pts_cam_frame, img)
        final_points = points[front_idx][outside_idx][:,:3]
        
        pcd = get_pointcloud(final_points, rgb, visualize=visualize)
        return pcd

    def store_segmented_pointcloud(self, start_idx=0, end_idx=1000):

        for i in range(start_idx, end_idx):
            pcd = self.paint_pointcloud(i)
            o3d.io.write_point_cloud(f"{self.dir_path}/dataset/segmented_pcd/{i-start_idx}.pcd", pcd)  

    def visualize_segmented_pcd(self, idx):
        pcd_load = o3d.io.read_point_cloud(f"{self.segmented_pcd}/{idx}.pcd")
        o3d.visualization.draw_geometries([pcd_load])

    def load_pcds(self):
        pcds = []
        pcd_data = os.listdir(self.painted_pcd_rgb)
        pcd_data.sort()
        for idx in range(len(pcd_data)):
            pcd = o3d.io.read_point_cloud(f"{self.painted_pcd_rgb}/{pcd_data[idx]}")
            pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
            pcds.append(pcd_down)
        return pcds
    
if __name__ == "__main__":
    proj = BEV()
    proj.visualize_segmented_pcd(625)
   