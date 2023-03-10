import numpy as np
import open3d as o3d
import os
from os.path import dirname, abspath
import matplotlib.pyplot as plt
import cv2
from utils import *
from calib import Calibration

class Projection():
    def __init__(self) -> None:
        self.dir_path = dirname(dirname(abspath(__file__)))
        self.calib_path = f"{self.dir_path}/dataset/calibration"
        self.img_dataset_path = f"{self.dir_path}/dataset/camera/images"
        self.pcd_dataset_path = f"{self.dir_path}/dataset/lidar/pcd_points"
        self.painted_pcd_rgb = f"{self.dir_path}/dataset/painted_points_rgb"
        self.calib = Calibration(self.calib_path)
        
        self.voxel_size = 0.05
        
        all_pcds = self.load_pcds()
        self.pcds_down = all_pcds

    def get_single_data(self, idx):
        pcd_data = os.listdir(self.pcd_dataset_path)
        pcd_data.sort()
        img_data = os.listdir(self.img_dataset_path)
        img_data.sort()

        pcd_load = o3d.io.read_point_cloud(f"{self.pcd_dataset_path}/{pcd_data[idx]}")
        points = np.asarray(pcd_load.points)
        img = cv2.imread(f"{self.img_dataset_path}/{img_data[idx]}")
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

    def store_painted_pointcloud(self):
        start_idx = 2500
        end_idx = 2600
        for i in range(start_idx, end_idx):
            pcd = self.paint_pointcloud(i)
            o3d.io.write_point_cloud(f"{self.dir_path}/dataset/painted_points_rgb/{i-start_idx}.pcd", pcd)  

    def visualize_painted_rgb_pcd(self, idx):
        pcd_load = o3d.io.read_point_cloud(f"{self.painted_pcd_rgb}/{idx}.pcd")
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
    
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target])

    def merge_pointcloud(self):
        source = self.pcds_down[5]
        target = self.pcds_down[6]
        target.estimate_normals()

        # point to plane ICP
        current_transformation = np.identity(4)
        print("2. Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. Distance threshold 0.02.")
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target, 0.02, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(result_icp)
        self.draw_registration_result(source, target, result_icp.transformation)

        voxel_radius = [0.04, 0.02, 0.01]
        max_iter = [50, 30, 14]
        current_transformation = np.identity(4)
        print("3. Colored point cloud registration")
        for scale in range(3):
            iter = max_iter[scale]
            radius = voxel_radius[scale]
            print([iter, radius, scale])

            print("3-1. Downsample with a voxel size %.2f" % radius)
            source_down = source.voxel_down_sample(radius)
            target_down = target.voxel_down_sample(radius)

            print("3-2. Estimate normal.")
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

            print("3-3. Applying colored point cloud registration")
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, current_transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                relative_rmse=1e-6,
                                                                max_iteration=iter))
            current_transformation = result_icp.transformation
            print(result_icp)
        self.draw_registration_result(source, target, result_icp.transformation)

if __name__ == "__main__":
    proj = Projection()
    proj.merge_pointcloud()
