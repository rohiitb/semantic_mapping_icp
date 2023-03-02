# semantic_mapping_icp

This is naive implementation of PointPainting where any image segmentation network can be used for 3D point cloud segmentaiton where each point is labelled with a class. In this case, SegFormer Transformer network is used to perform semantic segmentation on the images and the corresponding points in the pointcloud obtained from the LiDAR is classified. Several ICP trials were also carried out to merge the pointclouds to obtain 3D reconstruction of the scene. For each pointcloud, the Bird's eye view is also obtained.
<br> For demonstration purposes, [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/) is used.

## Results

### Semantically Segmented Pointcloud



### Bird's eye view

### Semantic segmentation results

### Installation

## File structure


