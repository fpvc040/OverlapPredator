import numpy as np
import copy
import time
import open3d as o3d

src_pcd_before = o3d.io.read_point_cloud("assets/1.pcd") 
tgt_pcd_before = o3d.io.read_point_cloud("assets/2.pcd")
print("point clouds read")
tsfm = [[-0.26057848,-0.96492304,-0.03197459,-0.00271199],[0.96476801,-0.2590038,-0.04625709,-0.00284887],[0.03635299,-0.04290167,0.9984177,-0.02050801],[ 0.,0.,0.,1.]]
print(tsfm)
src_pcd_after = copy.deepcopy(src_pcd_before)
src_pcd_after.transform(tsfm)
pcd_combined = o3d.geometry.PointCloud()
pcd_combined += tgt_pcd_before
pcd_combined += src_pcd_after
o3d.io.write_point_cloud("merged.pcd", pcd_combined)
o3d.io.write_point_cloud("transformed_tsfm.pcd", src_pcd_after)