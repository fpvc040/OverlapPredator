# examples/Python/Basic/pointcloud.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("/media/taru/8C3AC36A3AC3503E/Point-Cloud-Data/PCD/merged_1_2.pcd")
    o3d.visualization.draw_geometries([pcd])
