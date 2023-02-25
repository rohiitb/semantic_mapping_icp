import numpy as np
import open3d as o3d
import os

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

  
    
if __name__ == "__main__":
    pass