import argparse
import os
import sys
from tqdm import tqdm
import numpy as np
import torch

sys.path.append('../ubiphysio/')
from common.skeleton import Skeleton
from common.quaternion import qbetween_np, qrot_np, quaternion_to_cont6d_np, qmul_np, qinv_np
from utils.paramUtil import archive_chain, archive_offsets

# Positions (batch, joint_num, 3)
def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    
    # Calculate Scale Ratio as the ratio of leg lengths
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / src_leg_len

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    # Inverse Kinematics
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_idx)

    # Forward Kinematics
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints

def process_file(positions: np.array):
    """
    Process a single file's xyz coordinates to extract features.
    Args:
        positions: (seq_len, joints_num, 3) xyz coordinates
    """

    # Part 1: Normalization
    # Uniform Skeleton
    positions = uniform_skeleton(positions, tgt_offsets)

    # Put on floor
    floor_height = positions[:, :, 1].min(axis=1, keepdims=True)
    positions[:, :, 1] -= floor_height

    # XZ at origin
    pos_init = positions[0]
    root_pos_init = pos_init[0]
    root_pos_init_xz = root_pos_init * np.array([1, 0, 1])
    positions = positions - root_pos_init_xz

    # Face Z+ initially
    r_hip, l_hip, r_sdr, l_sdr = face_joint_idx
    across1 = pos_init[r_hip] - pos_init[l_hip]
    across2 = pos_init[r_sdr] - pos_init[l_sdr]
    across = across1 + across2
    across = across / np.linalg.norm(across)

    forward_init = np.cross(np.array([[0, 1, 0]]), across)
    forward_init = forward_init / np.linalg.norm(forward_init)

    forward_target = np.array([[0, 0, 1]])  # Target facing unit vector Z+
    forward_rot_quat = qbetween_np(forward_init, forward_target)  # Quaternion from initial to target facing
    forward_rot_quat = np.ones(positions.shape[:-1] + (4,)) * forward_rot_quat  # Expand to (seq_len, joint_num, 4)
    positions = qrot_np(forward_rot_quat, positions)  # Perform rotation

    global_positions = positions.copy()

    # Part 2: Feature Extraction
    # Foot Contact Detection
    def foot_detect(positions, thres):
        feet_l_h = positions[:-1, fid_l, 1]
        feet_r_h = positions[:-1, fid_r, 1]

        ignore_joints = [l_hand, r_hand]
        mask = np.ones(positions.shape[1], dtype=bool)
        mask[ignore_joints] = False
        contact_positions = positions[:, mask, :]
        contact_height = contact_positions[:-1, :, 1].min(axis=1, keepdims=True)

        feet_l_h -= contact_height
        feet_r_h -= contact_height

        feet_l = (feet_l_h < thres).astype(float)
        feet_r = (feet_r_h < thres).astype(float)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, 0.05)

    # Quaternion and Cartesian representation
    def get_cont6d_params(positions):
        skel = Skeleton(raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_idx, smooth_forward=True)

        # Quaternion to continuous 6D
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()

        # Root Linear Velocity
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)

        # Root Angular Velocity
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)

    def get_rifke(positions):
        # Local pose
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]

        # All pose face Z+
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    positions = get_rifke(positions)

    # Root height
    root_y = positions[:, 0, 1:2]

    # Root rotation and linear velocity
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    # Get Joint Rotation Representation
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    # Get Joint Rotation Invariant Position Representation
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    # Get Joint Velocity Representation
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    # Part 3: Biomechanical Features
    def calc_distance(idx1, idx2):
        positions1 = global_positions[:, idx1]
        positions2 = global_positions[:, idx2]
        vector = positions1 - positions2
        return np.sqrt(np.sum(np.square(vector), axis=1))

    def calc_angle4(idx1, idx2, idx3, idx4):
        position1 = global_positions[:, idx1]
        position2 = global_positions[:, idx2]
        position3 = global_positions[:, idx3]
        position4 = global_positions[:, idx4]
        vector1 = position2 - position1
        vector2 = position4 - position3

        dot_product = np.sum(vector1 * vector2, axis=1)
        cross_product = np.linalg.norm(np.cross(vector1, vector2), axis=1)
        return np.arctan2(cross_product, dot_product)

    def calc_plane_xz_angle(idx1, idx2, idx3):
        position1 = global_positions[:, idx1]
        position2 = global_positions[:, idx2]
        position3 = global_positions[:, idx3]
        vector1 = position1 - position2
        vector2 = position3 - position2

        normal_vector = np.cross(vector1, vector2)
        xz_normal = np.array([0, 1, 0])

        dot_product = np.sum(normal_vector * xz_normal, axis=1)
        cross_product = np.linalg.norm(np.cross(normal_vector, xz_normal), axis=1)
        return np.arctan2(cross_product, dot_product)

    def calc_angle(idx1, idx2, idx3):
        return calc_angle4(idx2, idx1, idx2, idx3)

    # General Features
    balance_up = calc_distance(l_hand, spine3) - calc_distance(r_hand, spine3) + calc_distance(l_forearm, spine3) - calc_distance(r_forearm, spine3)
    balance_down = calc_distance(l_foot, hip) - calc_distance(r_foot, hip) + calc_distance(l_leg, hip) - calc_distance(r_leg, hip)
    general_data = np.stack((balance_up, balance_down), axis=-1)

    # Angle Features
    angles = [
        calc_angle(l_sdr, spine3, spine2), calc_angle(r_sdr, spine3, spine2),
        calc_angle(spine3, spine2, spine1),
        calc_angle(spine2, spine1, hip),
        calc_angle(spine1, hip, l_upleg), calc_angle(spine1, hip, r_upleg),
        calc_angle(spine1, hip, l_leg), calc_angle(spine1, hip, r_leg),
        calc_angle4(l_sdr, r_sdr, l_upleg, r_upleg),
        calc_angle(l_upleg, l_leg, l_foot), calc_angle(r_upleg, r_leg, r_foot),
        calc_plane_xz_angle(l_sdr, spine3, hip), calc_plane_xz_angle(r_sdr, spine3, hip),
        calc_angle(l_sdr, hip, r_sdr),
        calc_angle(l_leg, l_foot, l_footend), calc_angle(r_leg, r_foot, r_footend)
    ]
    angles_data = np.stack(angles, axis=-1)

    # Distance Features
    distances = [
        calc_distance(l_sdr, hip), calc_distance(r_sdr, hip),
        calc_distance(hip, l_foot), calc_distance(hip, r_foot),
        calc_distance(l_hand, l_leg), calc_distance(l_hand, r_leg),
        calc_distance(r_hand, l_leg), calc_distance(r_hand, r_leg),
        calc_distance(l_hand, r_leg) / calc_distance(l_hand, l_leg),
        calc_distance(r_hand, l_leg) / calc_distance(r_hand, r_leg)
    ]
    distances_data = np.stack(distances, axis=-1)

    data = np.concatenate([data, general_data[:-1], angles_data[:-1], distances_data[:-1]], axis=-1)
    return data, global_positions

if __name__ == '__main__':
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Extract features from xyz coordinates")

    parser.add_argument('--joints_dir', type=str, default='/datax/zhonglx/archive/file2process', help='Directory of xyz coordinate data')
    parser.add_argument('--output_dir', type=str, default='/datax/zhonglx/archive/new_joint_vecs', help='Output directory for feature data')
    parser.add_argument('--outpos_dir', type=str, default='/datax/zhonglx/archive/new_joints', help='Output directory for new xyz coordinates')

    args = parser.parse_args()
    for directory in [args.output_dir, args.outpos_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    # Other parameters
    # Skeleton structure data
    raw_offsets = torch.from_numpy(archive_offsets)
    kinematic_chain = archive_chain
    # Sample data to determine tgt_offsets
    example_file = 'C01-000000-1.npy'
    # Key joints for feature calculation
    l_idx1, l_idx2 = 21, 22
    # [r_hip, l_hip, r_sdr, l_sdr] used for determining facing direction
    face_joint_idx = [20, 16, 8, 4]
    # Right/Left Foot (heels first)
    fid_r, fid_l = [22, 23], [18, 19]
    # Other joints
    r_hand, l_hand = 11, 7
    r_forearm, l_forearm = 10, 6
    r_upleg, l_upleg = 20, 16
    r_leg, l_leg = 21, 17
    r_foot, l_foot = 22, 18
    r_footend, l_footend = 23, 19
    r_sdr, l_sdr = 8, 4
    hip, spine1, spine2, spine3 = 0, 1, 2, 3
    num_joints = 24

    # Get offsets of target skeleton
    example_data = np.load(os.path.join(args.joints_dir, example_file))
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    # Process all files in the input directory
    frame_num = 0
    source_list = os.listdir(args.joints_dir)
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(args.joints_dir, source_file))
        try:
            features, new_joints = process_file(source_data)
            print(f'The size of feature is {features.shape}')
            print(f'The size of new joints is {new_joints.shape}')
            np.save(os.path.join(args.output_dir, source_file), features)
            np.save(os.path.join(args.outpos_dir, source_file), new_joints)
            frame_num += features.shape[0]
        except Exception as e:
            print(f"Error while processing {source_file}")
            print(e)

    print(f"Total clips: {len(source_list)}, Frames: {frame_num}, Duration: {frame_num / 60 / 60} min")
