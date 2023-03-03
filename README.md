# semantic_mapping_icp

This is naive implementation of PointPainting where any image segmentation network can be used for 3D point cloud segmentaiton where each point is labelled with a class. In this case, SegFormer Transformer network is used to perform semantic segmentation on the images and the corresponding points in the pointcloud obtained from the LiDAR is classified. Several ICP trials were also carried out to merge the pointclouds to obtain 3D reconstruction of the scene. For each pointcloud, the Bird's eye view is also obtained.
<br> For demonstration purposes, [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/) is used.

## Results

### Semantically Segmented Pointcloud
<img src="./results/pcd_gif.gif"  align="center"/>


### Bird's eye view
<img src="/github_fig/projected.png"  align="center" alt="Undistorted" width="400"/>

### Semantic segmentation results
<img src="/github_fig/projected.png"  align="center" alt="Undistorted" width="400"/>


### Installation



## File structure

    ├── seg SegFormer Folders
    ├── dataset    <--KITTI360
    |  ├── camera
    |  ├── calibration
    |  ├── lidar
    |  ├── painted_points_rgb
    |  ├── segmented_bev
    |  ├── segmented_imgs
    |  ├── segmented_pcd
    |  ├── whole_bev
    ├── code
    |  ├── utils.py 
    |  ├── calib.py
    |  ├── icp_trial_color_registration.py
    |  ├── icp_trial_global_registration.py
    |  ├── icp_trial_pose_graph_registration.py
    |  ├── segment_pcd.py
    |  ├── bev.py


