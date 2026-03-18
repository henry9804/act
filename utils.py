import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.v2 as transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import IPython
e = IPython.embed

active_joints = [25, 26, 27, 28, 29, 30, 31, 32, 33] # hardcode, TOCABI right arm + hand

color_transform = transforms.ColorJitter(brightness=0.5,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1,
                          )
random_crop = transforms.RandomCrop(size=(84,84))

def rotate_n_crop_transform(img, size, angle=None, top=None):
    if angle is None:
        angle = np.random.random() * 10 - 5
    if top is None:
        h, w = img.shape
        top_h = np.random.randint(0, 120)
        top_w = np.random.randint(0, 160)
        top = [top_h, top_w]
    
    img = transforms.functional.rotate(img, angle)
    img = transforms.functional.crop(img, *top, *size)

    return img


class EpisodicLeRobotDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, lerobot_dataset: LeRobotDataset, camera_names, n_obs_steps, chunk_size, state_key, action_key, norm_stats, img_aug=False, waypoint_indices=None):
        super(EpisodicLeRobotDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset = lerobot_dataset
        self.episodes = lerobot_dataset.meta.episodes
        self.stats = lerobot_dataset.meta.stats
        self.camera_names = camera_names
        self.n_obs_steps = n_obs_steps
        self.chunk_size = chunk_size
        self.state_key = state_key
        self.action_key = action_key
        self.norm_stats = norm_stats
        self.waypoint_indices = waypoint_indices  # dict: episode_id -> list
    
    def __len__(self):
        return len(self.episode_ids)
    
    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        frame_id = np.random.randint(self.episodes['dataset_from_index'][episode_id], self.episodes['dataset_to_index'][episode_id])
        data = self.dataset[frame_id]

        image_dict = dict()
        for cam_name in self.camera_names:
            if cam_name == "":  # pusht
                img = data[f'observation.image']
                # img = transforms.functional.resize(img, [480, 640])
                img = random_crop(img)
                image_dict[cam_name] = img
            elif cam_name.endswith('stereo'):
                left_img = data[f'observation.images.{cam_name[:-6]}left']
                right_img = data[f'observation.images.{cam_name[:-6]}right']
                left_img = transforms.functional.resize(left_img, [480, 640])
                right_img = transforms.functional.resize(right_img, [480, 640])
                image_dict[cam_name] = torch.cat([left_img, right_img], dim=-1) # width dimension
            else:
                img = data[f'observation.images.{cam_name}']
                img = transforms.functional.resize(img, [480, 640])
                image_dict[cam_name] = img
        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        if self.n_obs_steps == 1:
            image_data = torch.stack(all_cam_images, axis=1)
        else:
            image_data = torch.cat(all_cam_images, dim=1)

        # normalization (image is already normalized to 0-1)
        qpos_data = data[self.state_key]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = torch.flatten(qpos_data)  # handle n_obs_steps

        if self.waypoint_indices is not None:
            local_offset = frame_id - self.episodes['dataset_from_index'][episode_id]
            waypoints = np.array(self.waypoint_indices[episode_id]) - local_offset
            waypoints = waypoints[waypoints >= 0]

            dof = len(self.norm_stats["qpos_mean"])
            action_data = torch.zeros([self.chunk_size, dof])
            start_idx = 0
            for key_idx in waypoints:
                action_data[start_idx:key_idx+1] = torch.tensor([0.0, 0.0])
                # action_data[start_idx:key_idx+1] = self.dataset[key_idx+frame_id][self.state_key][-1]
                start_idx = key_idx
                if key_idx > self.chunk_size:
                    break
            
            ep_len = int(self.episodes['dataset_to_index'][episode_id]) - int(self.episodes['dataset_from_index'][episode_id])
            remaining = ep_len - local_offset
            is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
            if remaining < self.chunk_size:
                is_pad[remaining:] = True
        else:
            action_data = data[self.action_key]
            is_pad = data['action_is_pad']

        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return image_data, qpos_data, action_data, is_pad

def compute_waypoints(lerobot_dataset: LeRobotDataset, num_episodes, state_key, err_threshold=0.05, pos_only=False):
    """Precompute waypoints per episode using greedy_waypoint_selection.

    Returns:
        waypoints: dict mapping episode_id -> list (waypoint indices)
        wp_mean: np.array (action_dof,)
        wp_std:  np.array (action_dof,)
    """
    from waypoint_extraction.extract_waypoints import greedy_waypoint_selection

    episodes_meta = lerobot_dataset.meta.episodes
    waypoints = {}
    all_wp_data = []

    for ep_id in range(num_episodes):
        from_idx = int(episodes_meta['dataset_from_index'][ep_id])
        to_idx = int(episodes_meta['dataset_to_index'][ep_id])

        # use only the current timestep (index -1)
        states = np.stack([lerobot_dataset[i][state_key][-1].numpy()
                           for i in range(from_idx, to_idx)])

        wp_indices = greedy_waypoint_selection(
            actions=states,
            gt_states=states,
            err_threshold=err_threshold,
            geometry=True,
            pos_only=pos_only,
        )
        waypoints[ep_id] = wp_indices

        ep_waypoints = states[wp_indices]
        all_wp_data.append(ep_waypoints)

    all_wp_data = np.concatenate(all_wp_data, axis=0)
    wp_mean = all_wp_data.mean(axis=0).astype(np.float32)
    wp_std = all_wp_data.std(axis=0).clip(1e-2).astype(np.float32)

    return waypoints, wp_mean, wp_std


def load_lerobot_data(config: dict, chunk_size, batch_size_train, batch_size_val):
    dataset_id = config['dataset_id']
    camera_names = config['camera_names']
    num_episodes = config['num_episodes']
    n_obs_steps = config.get('n_obs_steps', 1)
    state_key = config.get('state_key', 'observation.state')
    action_key = config.get('action_key', 'action')
    dt = 1 / config.get('fps', 30)

    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # make delta_timestamps
    ds_meta = LeRobotDatasetMetadata(dataset_id)
    use_computed_waypoints = action_key == 'waypoint' and action_key not in ds_meta.stats
    delta_timestamps = {
        state_key: [dt*i for i in range(1 - n_obs_steps, 1)],
    }
    if not use_computed_waypoints:
        delta_timestamps[action_key] = [dt*i for i in range(chunk_size)]
    for cname in camera_names:  # check for camera name availability
        if cname == "":
            assert f'observation.image' in ds_meta.camera_keys, f'image is not included in {dataset_id} dataset'
            delta_timestamps['observation.image'] = [dt*i for i in range(1 - n_obs_steps, 1)]
        else:
            assert f'observation.images.{cname}' in ds_meta.camera_keys, f'{cname} image is not included in {dataset_id} dataset'
            delta_timestamps[f'observation.images.{cname}'] = [dt*i for i in range(1 - n_obs_steps, 1)]
    
    # obtain normalization stats for qpos and action
    norm_stats = {"qpos_mean": ds_meta.stats[state_key]['mean'].astype(np.float32),
                  "qpos_std": ds_meta.stats[state_key]['std'].astype(np.float32)}
    
    # construct dataset and dataloader
    lerobot_dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    waypoint_indices = None
    if use_computed_waypoints:
        print(f"'waypoint' key not found in dataset — computing waypoints via greedy_waypoint_selection ...")
        err_threshold = config.get('waypoint_err_threshold', 0.05)
        pos_only = config.get('waypoint_pos_only', False)
        waypoint_indices, wp_mean, wp_std = compute_waypoints(
            lerobot_dataset, num_episodes, state_key, err_threshold=err_threshold, pos_only=pos_only
        )
        norm_stats["action_mean"] = wp_mean
        norm_stats["action_std"] = wp_std
    else:
        norm_stats["action_mean"] = ds_meta.stats[action_key]['mean'].astype(np.float32)
        norm_stats["action_std"] = ds_meta.stats[action_key]['std'].astype(np.float32)

    train_dataset = EpisodicLeRobotDataset(train_indices, lerobot_dataset, camera_names, n_obs_steps, chunk_size, state_key, action_key, norm_stats, waypoint_indices=waypoint_indices)
    val_dataset = EpisodicLeRobotDataset(val_indices, lerobot_dataset, camera_names, n_obs_steps, chunk_size, state_key, action_key, norm_stats, waypoint_indices=waypoint_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=0)

    norm_stats["example_qpos"] = lerobot_dataset[0][state_key]

    return train_dataloader, val_dataloader, norm_stats, 'sim' in dataset_id


class EpisodicJointDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats, img_aug=False):
        super(EpisodicJointDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
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
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = transforms.functional.resize(left_img, [480, 640])
                    right_img = transforms.functional.resize(right_img, [480, 640])
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    img = transforms.functional.resize(img, [480, 640])
                    image_dict[cam_name] = img
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


def load_joint_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val, img_aug=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_joint_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicJointDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
    val_dataset = EpisodicJointDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


class EpisodicPoseDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, chunk_size, norm_stats, img_aug):
        super(EpisodicPoseDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.chunk_size = chunk_size
        self.norm_stats = norm_stats
        self.img_aug = img_aug
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
            episode_len = root['/observations/ee_pose_global'].shape[0] - 120  # hardcode for TOCABI data, do not train moving to ready pose
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/ee_pose_global'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if cam_name.endswith('stereo'):
                    left_img = root[f'/observations/images/{cam_name[:-6]}left'][start_ts]
                    right_img = root[f'/observations/images/{cam_name[:-6]}right'][start_ts]
                    left_img = transforms.functional.to_pil_image(left_img)
                    right_img = transforms.functional.to_pil_image(right_img)
                    if self.img_aug:
                        angle = np.random.random() * 10 - 5
                        top_h = np.random.randint(0, 120)
                        top_w = np.random.randint(0, 160)
                        left_img = color_transform(left_img)
                        left_img = rotate_n_crop_transform(left_img, [480, 640], angle, (top_h, top_w))
                        right_img = color_transform(right_img)
                        right_img = rotate_n_crop_transform(right_img, [480, 640], angle, (top_h, top_w))
                    left_img = transforms.functional.resize(left_img, [480, 640])
                    right_img = transforms.functional.resize(right_img, [480, 640])
                    image_dict[cam_name] = np.concatenate([left_img, right_img], axis=1) # width dimension
                else:
                    img = root[f'/observations/images/{cam_name}'][start_ts]
                    img = transforms.functional.to_pil_image(img)
                    if self.img_aug:
                        img = color_transform(img)
                        img = rotate_n_crop_transform(img)
                    img = transforms.functional.resize(img, [480, 640])
                    image_dict[cam_name] = img
            # get all actions after and including start_ts
            action = root['/ee_action_global'][start_ts:min(start_ts+self.chunk_size, episode_len)]
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
            qpos = root['/observations/ee_pose_global'][()]
            action = root['/ee_action_global'][()]
        all_qpos_data.append(torch.from_numpy(qpos).float())
        all_action_data.append(torch.from_numpy(action).float())
    all_qpos_data = torch.cat(all_qpos_data)
    all_action_data = torch.cat(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=0).numpy()
    action_std = all_action_data.std(dim=0).numpy()
    action_std = action_std.clip(1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=0).numpy()
    qpos_std = all_qpos_data.std(dim=0).numpy()
    qpos_std = qpos_std.clip(1e-2, np.inf) # clipping

    # do not normalize 9D roation & binary gripper state
    num_eef = all_qpos_data.shape[1] // 13
    for i in range(num_eef):
        action_mean[i*13:i*13+9] = 0.0
        action_mean[i*13+12:i*13+13] = 0.0
        qpos_mean[i*13:i*13+9] = 0.0
        qpos_mean[i*13+12:i*13+13] = 0.0
        action_std[i*13:i*13+9] = 1.0
        action_std[i*13+12:i*13+13] = 1.0
        qpos_std[i*13:i*13+9] = 1.0
        qpos_std[i*13+12:i*13+13] = 1.0

    stats = {"action_mean": action_mean, "action_std": action_std,
             "qpos_mean": qpos_mean, "qpos_std": qpos_std,
             "example_qpos": qpos}

    return stats


def load_pose_data(dataset_dir, num_episodes, camera_names, chunk_size, batch_size_train, batch_size_val, img_aug=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_pose_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicPoseDataset(train_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
    val_dataset = EpisodicPoseDataset(val_indices, dataset_dir, camera_names, chunk_size, norm_stats, img_aug)
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
