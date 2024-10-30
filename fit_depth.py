'''
for each frame generate a depth map using depth estimation model
then use linear regression to fit onto existing ground truth depth map
'''

import cv2 as cv
import os
import torch
import numpy as np
import json
from depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error
from helper_functions import *

def generate_depth_map_prediction(path='', visualize=False):
    if not path:
        print('data path not given')
        return
    json_data = get_transform(path,f'{path}/img')
    cv.namedWindow('img', cv.WINDOW_NORMAL) 
    cv.namedWindow('predicted', cv.WINDOW_NORMAL)
    frames = []
    i = 0
    for frame in json_data['frames']:
        i+=1
        print(f'current frame: {i}')
        # if i < 31:
        #     continue
        
        image_path = f'{path}/{frame["file_path"]}'
        depth_file_path = path + '/' +  frame['file_path'].split('.')[0] + '_depth_16.png'
        
        # extrinsic = np.linalg.inv(transform_matrix)
        depth_gt = cv.imread(depth_file_path, cv.IMREAD_UNCHANGED)
        image = cv.imread(image_path)
        h,w = image.shape[:2]
        # image = cv.resize(image,(h//2,w//2))
        # depth_gt = cv.resize(depth_gt, (h//2,w//2))
        
        depth = model.infer_image(image,FINAL_HEIGHT) # HxW depth map in meters in numpy
        depth = (depth*1000).astype(np.uint16)
        mask1 = depth_gt>2000
        mask2 = depth_gt<10000
        mask = np.bitwise_and(mask1, mask2)
        # # Fit the model
        # temp = depth_gt[mask]
        frame['gt_points'] = np.count_nonzero(mask)
        if np.count_nonzero(mask) < 1000:
            continue
        regression_model.fit(depth[mask].reshape(-1,1), depth_gt[mask].reshape(-1,1))
        
        prediction = regression_model.predict(depth.reshape(-1,1))
        mse = mean_squared_error(depth_gt[mask].reshape(-1,1), prediction[mask.reshape(-1,1)])
        # print(f'number of gt: {np.count_nonzero(mask)}')
        print(f'MSE score: {mse}')
        prediction = prediction.reshape(depth.shape)
        prediction_path = path + '/' +  frame['file_path'].split('.')[0] + '_predicted_16.png'
        
        frame['transform_matrix'] = frame['transform_matrix'].tolist()
        frames.append(frame)
        cv.imwrite(prediction_path, prediction.astype(np.uint16))
        if visualize:
            cv.imshow('predicted', prediction/np.max(prediction))
            cv.imshow('img', image)
        k = cv.waitKey(1)
        if k == 27:
            break
    cv.destroyAllWindows()
    json_data['frames'] = frames
    with open(f'{path}/transforms_colmap.json', 'w+') as f:
        json.dump(json_data, f, indent=2)


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
FINAL_HEIGHT=518

encoder = 'vitl' # or 'vits', 'vitb'
dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
max_depth = 20 # 20 for indoor model, 80 for outdoor model

model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
model.load_state_dict(torch.load('depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
model = model.cuda()
regression_model = LinearRegression()

if __name__ == '__main__':
    path = 'dataset/scene_kitchen'
    generate_depth_map_prediction(path=path)