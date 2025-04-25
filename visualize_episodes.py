import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
# from constants import DT
DT = 1/30

import IPython
e = IPython.embed

# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# STATE_NAMES = JOINT_NAMES + ["gripper"]
# STATE_NAMES = ["R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
#                "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"]
STATE_NAMES = ["Waist1_Joint", "Waist2_Joint", "Upperbody_Joint", "Neck_Joint", "Head_Joint",
               "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
               "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint",
               "hand"]

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        action = root['/action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, action, image_dict

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    if episode_idx == -1:
        episode_idx_list = range(len([fname for fname in os.listdir(dataset_dir) if fname.endswith(".hdf5")]))
    else:
        episode_idx_list = [episode_idx]
    
    # fig_all, axs_all = plt.subplots(1, 1, figsize=(26, 13))
    # fig_all, axs_all = plt.subplots(13, 1, figsize=(13, 26))
    for idx in episode_idx_list:
        dataset_name = f'episode_{idx}'

        qpos, qvel, action, image_dict = load_hdf5(dataset_dir, dataset_name)
        save_videos(image_dict, DT, video_path=os.path.join(dataset_dir, dataset_name + '_video.mp4'))
        # visualize_joints_all(qpos, axs_all)
        visualize_joints(qpos, action, plot_path=os.path.join(dataset_dir, dataset_name + '_qpos.svg'))
        # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back
    # plt.show()


def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {STATE_NAMES[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_joints_all(qpos_list, axs, ylim=None):
    qpos = np.array(qpos_list) # ts, dim
    num_ts, num_dim = qpos.shape

    # plot joint state
    # for dim_idx in range(num_dim):
    #     ax = axs[dim_idx]
    #     ax.plot(qpos[:, dim_idx])
    #     ax.set_title(f'Joint {dim_idx}: {STATE_NAMES[dim_idx]}')
    ax[0].plot(qpos[:, 4])
    ax.set_title(f'{STATE_NAMES[4]}')

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('-i', '--episode_idx', action='store', type=int, help='Episode index.', default=-1)
    main(vars(parser.parse_args()))
