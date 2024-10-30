import numpy as np
import os
import cv2 as cv

def point_cloud_to_voxel_with_density(points, points_color, voxel_size):
    voxel_indices = np.floor(points / voxel_size).astype(int)
    unique_indices, counts = np.unique(voxel_indices, axis=0, return_counts=True)
    voxel_array = np.hstack((unique_indices, counts[:, np.newaxis]))
    unique_indices, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    r_sums, g_sums, b_sums = np.zeros(len(unique_indices)), np.zeros(len(unique_indices)), np.zeros(len(unique_indices))

    np.add.at(r_sums, inverse_indices, points_color[:, 0])
    np.add.at(g_sums, inverse_indices, points_color[:, 1])
    np.add.at(b_sums, inverse_indices, points_color[:, 2])
    avg_r, avg_g, avg_b = r_sums / counts, g_sums / counts, b_sums / counts

    voxel_color = np.hstack((avg_r[:, np.newaxis], avg_g[:, np.newaxis], avg_b[:, np.newaxis]))
    return voxel_array, voxel_color

def read_depth_image(file_path):
    depth_image = cv.imread(file_path, cv.IMREAD_UNCHANGED)
    if depth_image is None or depth_image.dtype != np.uint16:
        raise ValueError(f"Invalid depth image at {file_path}")
    return depth_image

def read_rgb_image(file_path):
    rgb_image = cv.imread(file_path, cv.IMREAD_COLOR)
    if rgb_image is None:
        raise ValueError(f"Invalid RGB image at {file_path}")
    return cv.cvtColor(rgb_image, cv.COLOR_BGR2RGB).astype(np.float32)/255

def depth_to_point_cloud(depth_map, rgb_image, K, extrinsic_matrix):
    """
    Convert a depth map and RGB image to a colored point cloud.
    
    Parameters:
    depth_map (np.ndarray): 2D array containing depth values.
    rgb_image (np.ndarray): 2D array (H, W, 3) containing RGB values.
    K (np.ndarray): 3x3 camera intrinsic matrix.
    camera_transform (np.ndarray): 4x4 camera extrinsic transformation matrix.
    
    Returns:
    np.ndarray: 3D point cloud (Nx6), where each point has (X, Y, Z, R, G, B).
    """

    DOWNSCALE = 32
    height, width = depth_map.shape
    height = height//DOWNSCALE 
    width = width//DOWNSCALE
    depth_map = cv.resize(depth_map, (width, height))
    rgb_image = cv.resize(rgb_image, (width, height))
    fx, fy = K[0, 0]//DOWNSCALE, K[1, 1]//DOWNSCALE
    cx, cy = K[0, 2]//DOWNSCALE, K[1, 2]//DOWNSCALE

    # Generate the pixel coordinates (u,  v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert pixel coordinates to camera coordinates (Xc, Yc, Zc)
    Zc = depth_map.astype(np.float32)/1000
    Xc = (u - cx) * Zc / fx
    Yc = (v - cy) * Zc / fy

    # Stack into a (3, N) array of camera coordinates
    points_camera = np.vstack((Xc.flatten(), Yc.flatten(), Zc.flatten(), np.ones_like(Zc.flatten())))
    # camera_to_world_transform = np.linalg.inv(extrinsic_matrix)
    # Apply the camera extrinsic transformation (4x4 matrix) to get world coordinates
    points_world = extrinsic_matrix @ points_camera

    # Get corresponding RGB colors
    colors = rgb_image.reshape(-1, 3)

    # Concatenate 3D points with their corresponding colors (Nx6)
    
    return points_world[:3, :].T, colors
def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])
      
def get_transform(TEXT_FOLDER, IMAGE_FOLDER):
    SKIP_EARLY = 0
    cameras, camera_id = get_camera_params(TEXT_FOLDER)
    with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
        AABB_SCALE = 4
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        if len(cameras) == 1:
            camera = cameras[camera_id]
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": AABB_SCALE,
                "frames": [],
            }
        else:
            out = {
                "frames": [],
                "aabb_scale": AABB_SCALE
            }
    

        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY*2:
                continue
            if  i % 2 == 1:
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = f'img/{"_".join(elems[9:])}'
    
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                # c2w = np.linalg.inv(m)
    
                frame = {"file_path":name,"transform_matrix": m}
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                out["frames"].append(frame)
        return out

def parse_points3D(filename):
    points = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue  # Skip comment lines
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # Skip lines without enough data
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            points.append((x, y, z, r, g, b))
    xyz = np.array([[p[0], p[1], p[2]] for p in points])
    rgb = np.array([[p[3] / 255.0, p[4] / 255.0, p[5] / 255.0] for p in points])
    return xyz,rgb

def get_camera_params(TEXT_FOLDER):
    cameras = {}
    with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
        camera_angle_x = np.pi / 2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            camera["camera_angle_x"] = np.arctan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = np.arctan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / np.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / np.pi

            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera
    return cameras, camera_id
