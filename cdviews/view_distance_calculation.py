import os
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
import pandas as pd
import argparse
from qa_utils import load_and_update


def extract_camera_pose(pose_matrix, return_orientation=False):
    """Extract camera position and orientation as quaternion from pose matrix."""
    position = pose_matrix[:3, 3]
    orientation = R.from_matrix(pose_matrix[:3, :3])
    quaternion = orientation.as_quat()
    if return_orientation:
        return position, orientation
    return position, quaternion


def camera_distance(pose1, pose2, position_weight=1.0, orientation_weight=1.0):
    """
    Calculate a distance metric between two camera poses.
    Consider position and orientation (using quaternions).
    """
    pos1, quat1 = pose1
    pos2, quat2 = pose2

    # Position distance
    pos_distance = np.linalg.norm(pos1 - pos2)

    # Orientation distance (angle between quaternions)
    quat_dot_product = np.abs(np.dot(quat1, quat2))
    orientation_distance = 2 * np.arccos(np.clip(quat_dot_product, -1.0, 1.0))

    # Combine distances with weights
    return (position_weight * pos_distance +
            orientation_weight * orientation_distance)


def calculate_image_distance_(image_poses):
    # Extract camera poses
    poses = {img: extract_camera_pose(pose) for img, pose in image_poses.items()}

    # Create a list of images
    image_names = list(poses.keys())

    # Calculate pairwise distances
    n = len(image_names)
    distance_df = pd.DataFrame(np.zeros((n, n)), index=image_names, columns=image_names)
    for i in range(n):
        for j in range(i + 1, n):
            dist = camera_distance(poses[image_names[i]], poses[image_names[j]])
            distance_df.iloc[i, j] = dist
            distance_df.iloc[j, i] = dist
    return distance_df


def load_pose(filename):
    lines = open(filename).read().splitlines()
    print(filename)
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return np.asarray(lines).astype(np.float32)


def calculate_view_distance(scene_id, args):
    pose_path = os.path.join(args.image_folder, scene_id, 'pose')
    pose_file_list = [i for i in os.listdir(pose_path) if i[-4:] == '.txt']
    pose_dict = {}
    for pose_file in pose_file_list:
        pose_file_ = os.path.join(pose_path, pose_file)
        pose_info = load_pose(pose_file_)
        image_name = '{}.jpg'.format(pose_file[:-4])
        pose_dict[image_name] = pose_info
    distance_df = calculate_image_distance_(pose_dict)
    return distance_df


def save_view_distance(args):
    scene_list = [i for i in os.listdir(args.image_folder) if i[:5] == 'scene']
    for scene_id in tqdm(scene_list):
        distance_df = calculate_view_distance(scene_id, args)
        distance_df.to_csv(os.path.join(args.view_distance_folder, '{}.csv'.format(scene_id)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", type=str, default="../cfgs/QA.yaml")
    args = parser.parse_args()

    args = load_and_update(args)
    save_view_distance(args)

