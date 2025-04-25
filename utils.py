import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import IPython
e = IPython.embed

active_joints = [25, 26, 27, 28, 29, 30, 31, 32, 33] # hardcode, TOCABI right arm + hand

class EpisodicJointDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats):
        super(EpisodicJointDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root['/action'].shape[0] - 120    # hardcode, do not train moving to ready pose
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts, active_joints]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            action = root['/action'][start_ts:min(start_ts+self.chunk_size, episode_len), active_joints]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = transforms.Resize((480, 1280))(image_data) # hardcode for TOCABI stereo data

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_joint_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][:,active_joints]
            action = root['/action'][:,active_joints]
        all_qpos_data.append(torch.from_numpy(qpos[:,:-1]))   # do not normalize binary gripper state
        all_action_data.append(torch.from_numpy(action[:,:-1]))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = np.zeros_like(action[0], dtype=np.float32)
    action_mean[:-1] = all_action_data.mean(dim=0, keepdim=True)
    action_std = np.ones_like(action[0], dtype=np.float32)
    action_std[:-1] = all_action_data.std(dim=0, keepdim=True)
    action_std = action_std.clip(1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = np.zeros_like(qpos[0], dtype=np.float32)
    qpos_mean[:-1] = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std = np.ones_like(qpos[0], dtype=np.float32)
    qpos_std[:-1] = all_qpos_data.std(dim=0, keepdim=True)
    qpos_std = qpos_std.clip(1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}

    return stats


def load_joint_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_joint_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicJointDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats)
    val_dataset = EpisodicJointDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class EpisodicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats):
        super(EpisodicPoseDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            episode_len = root['/observations/ee_pose_rel'].shape[0] - 120  # hardcode for TOCABI data, do not train moving to ready pose
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/ee_pose_rel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            action = root['/ee_action_rel'][start_ts:min(start_ts+self.chunk_size, episode_len)]
            action_len, action_dof = action.shape

        self.is_sim = is_sim
        padded_action = np.zeros((self.chunk_size, action_dof), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        image_data = transforms.Resize((480, 1280))(image_data) # hardcode for TOCABI stereo data

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_pose_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/ee_pose_rel'][()]
            action = root['/ee_action_rel'][()]
        all_qpos_data.append(torch.from_numpy(qpos[:,9:12])) # do not normalize 9D roation & binary gripper state
        all_action_data.append(torch.from_numpy(action[:,9:12]))
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = np.zeros(13, dtype=np.float32)
    action_mean[9:12] = all_action_data.mean(dim=0).numpy()
    action_std = np.ones(13, dtype=np.float32)
    action_std[9:12] = all_action_data.std(dim=0).numpy()
    action_std = action_std.clip(1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = np.zeros(13, dtype=np.float32)
    qpos_mean[9:12] = all_qpos_data.mean(dim=0).numpy()
    qpos_std = np.ones(13, dtype=np.float32)
    qpos_std[9:12] = all_qpos_data.std(dim=0).numpy()
    qpos_std = qpos_std.clip(1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}

    return stats


def load_pose_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_pose_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicPoseDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats)
    val_dataset = EpisodicPoseDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
