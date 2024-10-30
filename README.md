# GDPT-Grounded Monocular Image Depth Estimation for Dense Point Cloud 3D Reconstruction

## What does this do?
This is my master's final project, it reads point cloud reconstruction output from COLMAP and use depth anything V2 to solve the issue where featureless area could not be reconstructed

## Installation
  ### download pretrained weights
  [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) download pretrained weights and put it under root folder of the project

  ### install pytorch with CUDA
  [Pytorch](https://pytorch.org/) it is recommended that you follow installation guide on pytorch to get optimal performance

  ### install required libraries with pip 
  > ```bash
  > pip install -r requirements.txt
  >   ```
## Usage
  run main.py under root foler for sample usage
## Create Custom Data
  [COLMAP](https://colmap.github.io/install.html) install COLMAP with given instruction, then run COLMAP and export data to txt file, put images and files under this structure:  
  📂 dataset  
  ├── 📂 Project_1  
  │   ├──  📂 img  
  │   ├──  📄 cameras.txt  
  │   ├──  📄 images.txt  
  │   ├──  📄 points3D.txt  
  └── 📂 Project_2  
      ├── 📂 img  
      ├── 📄 cameras.txt  
      ├── 📄 images.txt  
      └── 📄 points3D.txt  
