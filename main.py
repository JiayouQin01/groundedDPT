import json
import cv2 as cv
import numpy as np
import open3d as o3d
from get_depth_gt import generate_depth_map_gt
from fit_depth import generate_depth_map_prediction
from helper_functions import *

path = 'dataset/scene_kitchen'


generate_depth_map_gt(path=path, visualize = True)         #generate ground truth depth map from sparse point cloud
generate_depth_map_prediction(path=path, visualize = True) #generate fitted depth map
# depth map generation only need to run once for each data set

data = get_transform(path,f'{path}/img')
cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.namedWindow('depth', cv.WINDOW_NORMAL)

all_points, all_points_rgb = np.empty((0,3), dtype=np.float32), np.empty((0,3), dtype=np.float32)
all_pcd = None
intrinsic = np.array([[data['fl_x'], 0, data['cx']],
                      [0, data['fl_y'], data['cy']],
                      [0, 0, 1]])
for frame in data['frames']:
    rgb_file_path = f'{path}/{frame["file_path"]}'
    depth_file_path = rgb_file_path.replace(".jpg", "_predicted_16.png")
    print(depth_file_path)
    extrinsic_matrix = np.array(frame['transform_matrix'])
    
    R = extrinsic_matrix[:3, :3]  # 3x3 rotation matrix
    t = extrinsic_matrix[:3, 3]   # 3x1 translation vector
    
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R.T  # Transpose of the rotation matrix
    pose_matrix[:3, 3] = -np.dot(R.T, t)

    try:
        depth_map = read_depth_image(depth_file_path)
        rgb_image = read_rgb_image(rgb_file_path)
        cv.imshow('image', rgb_image)
        cv.imshow('depth', depth_map)
        if not np.any(depth_map):
            print('depth map invalid')
            continue

        points, colors = depth_to_point_cloud(depth_map, rgb_image, intrinsic, np.linalg.inv(extrinsic_matrix))
        # points /= 1000

        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(points)
        current_pcd.colors = o3d.utility.Vector3dVector(colors)
        all_points = np.concatenate((all_points, np.asarray(current_pcd.points)), axis=0)
        all_points_rgb = np.concatenate((all_points_rgb, np.asarray(current_pcd.colors)), axis=0)
        all_pcd = o3d.geometry.PointCloud()
        all_pcd.points = o3d.utility.Vector3dVector(all_points)
        all_pcd.colors = o3d.utility.Vector3dVector(all_points_rgb)
        
        # visualize point cloud per iteration for debugging purpose
        # o3d.visualization.draw_geometries([current_pcd])
        # o3d.visualization.draw_geometries([all_pcd])
        
        cv.waitKey(1)
    except ValueError as e:
        print(e)

cv.destroyAllWindows()

#show unprocessed point cloud
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(all_points[:, :3])
point_cloud.colors = o3d.utility.Vector3dVector(all_points_rgb)
o3d.visualization.draw_geometries([point_cloud.voxel_down_sample(0.1)])

#show processed point cloud
voxel, voxel_color = point_cloud_to_voxel_with_density(all_points, all_points_rgb, 0.1)
mask = voxel[:, 3] > 2
point_cloud.points = o3d.utility.Vector3dVector(voxel[mask][:, :3])
point_cloud.colors = o3d.utility.Vector3dVector(voxel_color[mask])
o3d.visualization.draw_geometries([point_cloud])