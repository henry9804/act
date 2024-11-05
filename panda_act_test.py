import os
import time
import numpy as np
import argparse
from matplotlib import pyplot as plt

import rospy
from sensor_msgs.msg import CompressedImage
from moveit_msgs.msg import DisplayRobotState

import h5py
from turbojpeg import TurboJPEG, TJPF_RGB

import pickle
import torch
from einops import rearrange
from policy import ACTPolicy
from constants import SIM_TASK_CONFIGS

import sys
USERNAME = os.getlogin()
sys.path.append(f'/home/{USERNAME}/catkin_ws/src/suhan_robot_model_tools')
from srmt.planning_scene import PlanningSceneLight
from srmt.kinematics import TRACIK
planning_scene = PlanningSceneLight(topic_name='planning_scene', base_link='/world')
tracik_panda = TRACIK('panda_link0', 'panda_hand')

FPS = 30
QUALITY = 30

def visualize_joints(qpos_list, action_list, output_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2, label3 = label_overwrite
    else:
        label1, label2, label3 = 'State', 'Action', 'Output'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(action_list)
    output = np.array(output_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: joint_{dim_idx}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    # plot act output
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(output[:, dim_idx], label=label3)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_ee(qpos_list, output_list, plot_path=None, ylim=None, label_overwrite=None):
    current = []
    target = []
    for qpos, output in zip(qpos_list, output_list):
        current_pos, current_quat = tracik_panda.forward_kinematics(qpos)
        target_pos, target_quat = tracik_panda.forward_kinematics(output)
        current.append(current_pos)
        target.append(target_pos)
    current = np.array(current)
    target = np.array(target)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(current[:,0], current[:,1], current[:,2], label='State')
    ax.plot(target[:,0], target[:,1], target[:,2], label='Output')
    ax.axis('equal')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.set_zlabel('z(m)')
    plt.show()


class TocabiAct:
    def __init__(self, args):
        is_eval = args['eval']
        ckpt_dir = args['ckpt_dir']
        policy_class = args['policy_class']
        onscreen_render = args['onscreen_render']
        task_name = args['task_name']
        batch_size_train = args['batch_size']
        batch_size_val = args['batch_size']
        num_epochs = args['num_epochs']
        self.temporal_agg = args['temporal_agg']

        task_config = SIM_TASK_CONFIGS[task_name]
        dataset_dir = task_config['dataset_dir']
        num_episodes = task_config['num_episodes']
        episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']

        ckpt_name = 'policy_best.ckpt'
        self.state_dim = task_config['model_dof']
        self.max_timesteps = int(episode_len * 2)

        policy_config = {'lr': args['lr'],
                        'num_queries': args['chunk_size'],
                        'kl_weight': args['kl_weight'],
                        'hidden_dim': args['hidden_dim'],
                        'dim_feedforward': args['dim_feedforward'],
                        'lr_backbone': 1e-5,
                        'backbone': 'resnet18',
                        'enc_layers': 4,
                        'dec_layers': 7,
                        'nheads': 8,
                        'camera_names': self.camera_names,
                        'state_dim': self.state_dim
                        }

        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        self.policy = ACTPolicy(policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')

        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        self.post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries']
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
        else:
            self.query_frequency = policy_config['num_queries']

        self.terminate = True
        self.t = 0
        self.all_actions = None

    def inference(self, qpos_numpy, curr_image):
        with torch.inference_mode():
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

            ### query policy
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            if self.temporal_agg:
                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t+1]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, self.t % self.query_frequency]

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)

            ### step the environment
            print(self.t, ': ', action)

            raw_final_action = self.all_actions[:, -1]
            final_action = self.post_process(raw_final_action.squeeze(0).cpu().numpy())

            self.t += 1
        
        return action, final_action


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
    tocabi_act = TocabiAct(vars(args))

    dataset_dir = '/media/embodied_ai/SSD2TB/act/data/real_panda_peg_in_hole'
    episode_index = 0
    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, f'episode_{episode_index}')
    print('done loading hdf5 file!')

    inf_pos = []
    max_timestep = len(qpos)
    for i in range(max_timestep):
        # calculate target pose
        qpos_numpy = np.array(qpos[i])
        torch_img = rearrange(image_dict['top'][i], 'h w c -> c h w')
        curr_image = np.stack([torch_img], axis=0)
        next_pos, goal_pos = tocabi_act.inference(qpos_numpy, curr_image)

        inf_pos.append(next_pos)

    visualize_joints(qpos, action, inf_pos, plot_path=os.path.join(vars(args)['ckpt_dir'], f'episode_{episode_index}_result_q.svg'))
    visualize_ee(qpos, inf_pos)#, plot_path=os.path.join(vars(args)['ckpt_dir'], f'episode_{episode_index}_result_ee.svg'))

'''
run with following args
python panda_act_test.py \
--policy_class ACT --kl_weight 10 --hidden_dim 512 --batch_size 8 \
--dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --temporal_agg \
--chunk_size 30 --task_name real_panda_peg_in_hole --ckpt_dir /media/embodied_ai/SSD2TB/act/ckpt/panda/peg_in_hole
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    args = parser.parse_args()
    main(args)
