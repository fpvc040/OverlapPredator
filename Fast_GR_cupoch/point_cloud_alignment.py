import cupoch as cph
import numpy as np
import copy
import time



def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    cph.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        cph.geometry.KDTreeSearchParamRadius(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_feature = cph.registration.compute_shot_feature(
        pcd_down, radius_feature,
        cph.geometry.KDTreeSearchParamRadius(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_feature


def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = cph.io.read_point_cloud("../assets/1.pcd")
    target = cph.io.read_point_cloud("../assets/2.pcd")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = cph.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        cph.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


if __name__ == "__main__":
    cph.utility.set_verbosity_level(cph.utility.Debug)

    voxel_size = 0.5  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
            prepare_dataset(voxel_size)

    start = time.time()
    result_fast = execute_fast_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    print(result_fast)
    print(result_fast.transformation)
    #draw_registration_result(source_down, target_down,
    #                         result_fast.transformation)
