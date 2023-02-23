import numpy as np
import open3d as o3d
import os

def conv_bin_to_pcd(ip_path, op_path):
    pcd_data = os.listdir(ip_path)
    pcd_data.sort()
    for i in range(len(pcd_data)):
        bin_pcd = np.fromfile(f"{ip_path}/{pcd_data[i]}", dtype=np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]

        # Convert to Open3D point cloud
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        # Save to whatever format you like

        o3d.io.write_point_cloud(f"{op_path}/{pcd_data[i].split('.')[0]}.pcd", o3d_pcd)        

if __name__ == "__main__":
    ip_path = "/home/rohiitb/semantic_mapping_icp/lidar_dataset/KITTI-360/data_3d_raw/2013_05_28_drive_0005_sync/velodyne_points/data"
    op_path = "/home/rohiitb/semantic_mapping_icp/dataset/pcd_points"
    conv_bin_to_pcd(ip_path, op_path)

