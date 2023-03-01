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
        pcd_data = os.listdir(self.segmented_pcd)
        pcd_data.sort()
        for idx in range(len(pcd_data)):
            pcd = o3d.io.read_point_cloud(f"{self.segmented_pcd}/{pcd_data[idx]}")
            pcds.append(pcd)
        return pcds
    
    def get_single_seg_data(self, idx):
        pcd_data = os.listdir(self.segmented_pcd)
        pcd_data.sort()
        pcd_load = o3d.io.read_point_cloud(f"{self.segmented_pcd}/{pcd_data[idx]}")
        points = np.asarray(pcd_load.points)
        return points
    
    def get_whole_bev(self, points, side_range=(-50,50), fwd_rng=(-50,50), res=0.05, height_rng=(-10,10.5)):
        '''
        Input: points in LiDAR frame (n,3), side_range (clipping the side values), fwd_rng (clipping the forward values), 
                res (resolution ...keep more for coarse pointcloud and less for fine), height_rng (clipping values in Z direction)
        Output: Bird's eye view image

        '''
        x_points = points[:,0]
        y_points = points[:,1]
        z_points = points[:,2]


        f_filter = np.logical_and((x_points > fwd_rng[0]), x_points < fwd_rng[1])
        s_filter = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filt = np.logical_and(f_filter, s_filter)
        idx = np.argwhere(filt).flatten()

        x_points = x_points[idx]
        y_points = y_points[idx]
        z_points = z_points[idx]

        x_img = (-y_points/res).astype(np.int32)
        y_img = (-x_points/res).astype(np.int32)

        x_img -= int(np.floor(side_range[0]/res))
        y_img += int(np.ceil(fwd_rng[1]/res))

        pixel_vals = np.clip(z_points, height_rng[0], height_rng[1])
        # print("here : ", pixel_vals)
        # raise
        pixel_vals = normalize_pixels(pixel_vals, height_rng[0], height_rng[1])

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = 1+int((side_range[1] - side_range[0])/res)
        y_max = 1+int((fwd_rng[1] - fwd_rng[0])/res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        im[y_img, x_img] = pixel_vals 

        plt.imshow(im, cmap="gray")
        plt.show()

    def get_seg_bev(self, points, side_range=(-50,50), fwd_rng=(-25,25), res=0.05, height_rng=(-2,0.5)):
        '''
        Input: points in LiDAR frame (n,3), side_range (clipping the side values), fwd_rng (clipping the forward values), 
                res (resolution ...keep more for coarse pointcloud and less for fine), height_rng (clipping values in Z direction)
        Output: Bird's eye view image

        '''
        x_points = points[:,0]
        y_points = points[:,1]
        z_points = points[:,2]


        f_filter = np.logical_and((x_points > fwd_rng[0]), x_points < fwd_rng[1])
        s_filter = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
        filt = np.logical_and(f_filter, s_filter)
        idx = np.argwhere(filt).flatten()

        x_points = x_points[idx]
        y_points = y_points[idx]
        z_points = z_points[idx]

        x_img = (-y_points/res).astype(np.int32)
        y_img = (-x_points/res).astype(np.int32)

        x_img -= int(np.floor(side_range[0]/res))
        y_img += int(np.ceil(fwd_rng[1]/res))

        pixel_vals = np.clip(z_points, height_rng[0], height_rng[1])
        # print("here : ", pixel_vals)
        # raise
        pixel_vals = normalize_pixels(pixel_vals, height_rng[0], height_rng[1])

        # INITIALIZE EMPTY ARRAY - of the dimensions we want
        x_max = 1+int((side_range[1] - side_range[0])/res)
        y_max = 1+int((fwd_rng[1] - fwd_rng[0])/res)
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        im = np.zeros((3, y_max, x_max)).astype(np.uint8)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        im[y_img, x_img] = pixel_vals 
        print("Image : ", im.max(), im.min())
        raise

        plt.imshow(im, cmap="gray")
        plt.show()





    
if __name__ == "__main__":
    proj = BEV()
    # proj.visualize_segmented_pcd(625)
    points_lid, _ = proj.get_single_data(0)
    print(points_lid.shape)
    proj.get_whole_bev(points_lid)

   