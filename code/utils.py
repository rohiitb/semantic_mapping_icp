import numpy as np
import open3d as o3d
import os
import copy

def conv_bin_to_pcd(ip_path, op_path):
    '''
    INPUT : ip_path - Provide the path to the bin files
    OUTPUT : op_path - Provide the path to the directory where you want to store pcd files
    '''
    pcd_data = os.listdir(ip_path)
    pcd_data.sort()
    for i in range(len(pcd_data)):
        bin_pcd = np.fromfile(f"{ip_path}/{pcd_data[i]}", dtype=np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]

        # Convert to Open3D point cloud
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        # Save to whatever format you like
        o3d.io.write_point_cloud(f"{op_path}/{pcd_data[i].split('.')[0]}.pcd", o3d_pcd)  

def lidar_2_cam(points, T_cam_2_lidar):
    '''
    INPUT : pointcloud points Shape(n,4), T_cam_2_lidar Shape(4,4)
    OUTPUT : pointcloud points Shape(n,4), 
             (bool) forward_pts_idx ....Points in front of camera
    '''
    points_transformed = (np.linalg.inv(T_cam_2_lidar) @ points.T).T   
    forward_pts_idx = points_transformed[:,2]>0
    points_transformed = points_transformed[forward_pts_idx]
    return points_transformed, forward_pts_idx

def proj_lidar_2_img(points, P_rect_cam, R_rect_cam, img_size):
    '''
    INPUT : pointcloud points Shape(n,4), P, R Shape(4,4), img_size tuple(1,2)
    OUTPUT : pointcloud pixels Shape(n,2)
             (bool) pts_cam_frame_idx ....Points inside the image
    '''
    pts_cam_frame = P_rect_cam @ R_rect_cam @ points.T
    pts_cam_frame = pts_cam_frame[:3] / pts_cam_frame[2]
    pts_cam_frame = pts_cam_frame[:2].T
    pts_cam_frame_idx = np.logical_and(pts_cam_frame[:,0] >= 0, np.logical_and(pts_cam_frame[:,1] >= 0, np.logical_and(pts_cam_frame[:,0] < img_size[0], pts_cam_frame[:,1] < img_size[1])))
    pts_cam_frame = pts_cam_frame[pts_cam_frame_idx]
    return pts_cam_frame, pts_cam_frame_idx

def make_pts_homogenous(points):
    '''
    Make points homogeneous Shape (n,4)
    '''
    points = np.hstack((points, np.ones(len(points)).reshape(-1,1)))
    return points

def get_rgb(uv, img):
    '''
    Get RGB values(normalized) to visualize
    '''
    pixels = np.int32(uv).T
    rgb = img[pixels[1],pixels[0]]
    return rgb/255.

def get_pointcloud(points, rgb, visualize=False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    if visualize: o3d.visualization.draw_geometries([pcd])
    return pcd


    
if __name__ == "__main__":
    pass