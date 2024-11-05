
# Iterative Closest-Points (ICP) for Vehicle Trajectory Estimation

This repository contains an implementation of the Iterative Closest-Points (ICP) algorithm to estimate the trajectory of a vehicle based on LiDAR point clouds from the KITTI dataset.

## Project Structure

- **data**: Directory containing necessary data files.
  - **ground_truth**: Stores the ground-truth transformations in `.npy` format, with each file providing true trajectory data.
  - **points_clouds**: Contains the point cloud scans in `.obj` format, representing the environment captured by the LiDAR sensor.
- **source**: Contains the Python code for the ICP implementation.
  - **iterative_closest_points.py**: Python script implementing the ICP algorithm from scratch.

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Trimesh (for loading point clouds in `.obj` format)

## Steps

1. **Prepare the Data**: Ensure that the `ground_truth` and `points_clouds` folders within the `data` directory contain the required `.npy` and `.obj` files.
2. **Run the Code**: Execute the `iterative_closest_points.py` script located in the `source` directory to process the point clouds and estimate the vehicle’s trajectory.
3. **Output and Visualization**: The script will generate a 3D plot comparing the estimated trajectory with the ground-truth, showing the vehicle's path in XYZ space.

## Usage

- To start, navigate to the `source` directory and run the `iterative_closest_points.py` script.
- The script will load the point cloud scans from the `points_clouds` folder, perform the ICP alignment for trajectory estimation, and output a 3D plot of the estimated trajectory compared to the ground-truth.

## Results

- **Trajectory Plot**: The resulting plot displays the estimated trajectory in 3D space alongside the ground-truth, allowing for visual assessment of alignment accuracy.
- **Accuracy Assessment**: Optionally, numerical metrics can be outputted for a quantitative comparison between the estimated and ground-truth transformations.

## Directory Structure

```
Iterative Closest-Points (ICP)/
│
├── data/
│   ├── ground_truth/         # Ground-truth data for trajectory verification (.npy format)
│   └── points_clouds/        # Point clouds in .obj format
│
├── source/
│   └── iterative_closest_points.py    # Python script implementing ICP algorithm
│
└── README.md                 # Project documentation
```
