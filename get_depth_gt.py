'''
generate the ground truth depth map and output to given folder
'''

import json
import numpy as np
import open3d as o3d
import cv2 as cv
from helper_functions import *


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])
    



def generate_depth_map_gt(path='',visualize=False):

    if not path:
        print('no data path given')
        return
    json_data = get_transform(path,f'{path}/img')

    points3D_file = f'{path}/points3D.txt'  # Path to your COLMAP points3D.txt file


    # Load the 3D points from points3D.txt and visualize it
    points, points_color = parse_points3D(points3D_file)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(points_color)

    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])

    #remove statistical outlier

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=12,
                                                    std_ratio=2.0)
    ind = np.array(ind, dtype=np.uint16)
    points = points[ind]
    points_color = points_color[ind]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(points_color)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])



    # ctr = vis.get_view_control()
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.namedWindow('depth', cv.WINDOW_NORMAL)

    width, height = int(json_data['w']), int(json_data['h'])
    fl_x, fl_y = json_data['fl_x'], json_data['fl_y']
    cx, cy = json_data['cx'], json_data['cy']
    for frame in json_data['frames']:
        extrinsic = np.array(frame['transform_matrix'])
        image_path = f'./{path}/{frame["file_path"]}'

        # Create pinhole camera parameters and set extrinsic
        camera_parameters = o3d.camera.PinholeCameraParameters()
        camera_parameters.extrinsic = extrinsic

        # Set intrinsic parameters (use dummy values or actual ones if available)
        K = np.array([
            [fl_x, 0, cx],
            [0, fl_y, cy],
            [0, 0, 1]
        ])
        
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        camera_coords = (extrinsic @ points_h.T).T
        
        # Project points onto image plane
        image_points_h = (K @ camera_coords[:, :3].T).T
        image_points = image_points_h[:, :2] / image_points_h[:, 2, None]
        depths = camera_coords[:, 2]
        
        # Initialize depth map
        depth_map = np.full((height, width), np.inf)
        
        # Populate the depth map
        for i in range(len(image_points)):
            u, v = int(image_points[i, 0]), int(image_points[i, 1])
            if 0 <= u < width and 0 <= v < height:
                depth_map[v, u] = min(depth_map[v, u], depths[i])
        
        # Replace inf values with 0 (or some other placeholder)
        depth_map[depth_map == np.inf] = 0
        depth_map[depth_map<0] = 0;
        # Convert depth map to millimeters and to 16-bit integers
        depth_map_mm = (depth_map * 1000).astype(np.uint16)
        # Save depth map as a 16-bit PNG
        depth_file_path = path + '/' +  frame['file_path'].split('.')[0] + '_depth_16.png'
        cv.imwrite(depth_file_path, depth_map_mm)
        print(f'saving to {depth_file_path}')
        
        image = cv.imread(image_path)
        cv.imshow('image', image)
        cv.imshow('depth', depth_map/np.max(depth_map))
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fl_x, fl_y, cx, cy)
        # camera_parameters.intrinsic = intrinsic
        # ctr.convert_from_pinhole_camera_parameters(camera_parameters,allow_arbitrary = True)

        # vis.poll_events()
        # vis.update_renderer()

        k = cv.waitKey(1)
        if k == 27:
            break

    cv.destroyAllWindows()
    # vis.run()
    # vis.destroy_window()

# Example usage
if __name__ == '__main__':

    path = 'dataset/scene_kitchen'
    generate_depth_map_gt(path=path, visualize=True)