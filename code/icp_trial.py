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

    def get_data(self):
        pass


    def proj_lid_2_image(pcd, image):
        pass
        

