import os
import time
import numpy as np
import argparse

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

FPS = 30
QUALITY = 30

joint_name = ["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
              "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
              "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
              "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
              "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
              "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
              "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
              "Neck_Joint", "Head_Joint",
              "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
              "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"]
hand_joint_name  = ["aa1" , "aa2" , "aa3" , "aa4" ,
                    "act1", "act2", "act3", "act4",
                    "dip1", "dip2", "dip3", "dip4",
                    "mcp1", "mcp2", "mcp3", "mcp4",
                    "pip1", "pip2", "pip3", "pip4"]
hand_open_state  = [0.0   , 0.0   , 0.0   , 0.0   ,
                    0.0550, 0.0550, 0.0550, 0.0550,
                    0.1517, 0.1517, 0.1517, 0.1517,
                    0.0687, 0.0687, 0.0687, 0.0687,
                    0.0088, 0.0088, 0.0088, 0.0088]
hand_close_state = [0.6   , 0.0   , 0.0   , 0.0   ,
                    0.6645, 0.6645, 0.6645, 0.6645,
                    0.8379, 0.8379, 0.8379, 0.8379,
                    0.8306, 0.8306, 0.8306, 0.8306,
                    0.9573, 0.9573, 0.9573, 0.9573]

joint_index = [12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
q_init = np.array([0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                   0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                   0.0, 0.0, 0.0,
                   0.3, 0.3, 1.5, -1.27, -1, 0.0, -1, 0.0,
                   0.0, 0.0,
                   -0.3, -0.9, -1.5, 1.57, 1.9, 0.0, 0.6, 0.0])

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
            target_qpos = action[:-1]
            target_hand = int(round(action[-1]))

            ### step the environment
            print(self.t, ': ', action)

            raw_final_action = self.all_actions[:, -1]
            final_action = self.post_process(raw_final_action.squeeze(0).cpu().numpy())

            self.t += 1
        
        return final_action


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

    rospy.init_node('tocabi_act', anonymous=True)

    img_l_pub = rospy.Publisher("/cam_LEFT/image_raw/compressed", CompressedImage, queue_size=10)
    img_r_pub = rospy.Publisher("/cam_RIGHT/image_raw/compressed", CompressedImage, queue_size=10)
    current_state_pub = rospy.Publisher('/current_state', DisplayRobotState, queue_size=10)
    goal_state_pub = rospy.Publisher('/goal_state', DisplayRobotState, queue_size=10)

    turbo_jpeg = TurboJPEG()

    dataset_dir = '/media/embodied_ai/SSD2TB/act/data/real_tocabi_pick_cup_cropped'
    episode_index = 50
    qpos, qvel, action, image_dict = load_hdf5(dataset_dir, f'episode_{episode_index}')
    print('done loading hdf5 file!')

    inf_time = []
    max_timestep = len(qpos)
    for i in range(max_timestep):
        start = time.time()

        # publish current state
        current_state = DisplayRobotState()
        current_state.state.joint_state.header.stamp = rospy.Time.now()
        current_state.state.joint_state.name = [*joint_name, *hand_joint_name]
        if int(qpos[i][-1]) == 1:
            current_state.state.joint_state.position = [*q_init, *hand_close_state]
        else:
            current_state.state.joint_state.position = [*q_init, *hand_open_state]
        for j, q in zip(joint_index, qpos[i][:-1]):
            current_state.state.joint_state.position[j] = q
        current_state_pub.publish(current_state)

        # publish images        
        left_img = image_dict['left'][i]
        compress_img_l = turbo_jpeg.encode(left_img, quality=QUALITY, pixel_format=TJPF_RGB)
        img_l_msg = CompressedImage()
        img_l_msg.header.stamp = rospy.Time.now()
        img_l_msg.format = 'jpeg'
        img_l_msg.data = compress_img_l
        img_l_pub.publish(img_l_msg)

        right_img = image_dict['right'][i]
        compress_img_r = turbo_jpeg.encode(right_img, quality=QUALITY, pixel_format=TJPF_RGB)
        img_r_msg = CompressedImage()
        img_r_msg.header.stamp = rospy.Time.now()
        img_r_msg.format = 'jpeg'
        img_r_msg.data = compress_img_r
        img_r_pub.publish(img_r_msg)
        
        # calculate and publish target pose
        qpos_numpy = np.array(qpos[i])
        left_torch_img = rearrange(left_img, 'h w c -> c h w')
        curr_image = np.stack([left_torch_img], axis=0)
        goal_pos = tocabi_act.inference(qpos_numpy, curr_image)

        goal_state = DisplayRobotState()
        goal_state.state.joint_state.header.stamp = rospy.Time.now()
        goal_state.state.joint_state.name = [*joint_name, *hand_joint_name]
        if int(round(goal_pos[-1])) == 1:
            goal_state.state.joint_state.position = [*q_init, *hand_close_state]
        else:
            goal_state.state.joint_state.position = [*q_init, *hand_open_state]
        for j, q in zip(joint_index, goal_pos[:-1]):
            goal_state.state.joint_state.position[j] = q
        goal_state_pub.publish(goal_state)

        if int(qpos[i][-1]) == 1 and int(round(goal_pos[-1])) == 0:
            print('hand open predicted!')
        if int(qpos[i][-1]) == 0 and int(round(goal_pos[-1])) == 1:
            print('hand close predicted!')

        end = time.time()
        inf_time.append(end-start)
        if (time.time() - start) < 1/FPS:
            time.sleep(1/FPS - (time.time() - start))

    print(f'inference time\nmin: {min(inf_time[1:]):.4f} avg: {np.mean(inf_time[1:]):.4f} max: {max(inf_time[1:]):.4f}')

'''
run with following args
python real_tocabi_act_test.py \
--policy_class ACT --kl_weight 10 --hidden_dim 512 --batch_size 8 \
--dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --temporal_agg \
--chunk_size 30 --task_name real_tocabi_pick_cup --ckpt_dir /media/embodied_ai/SSD2TB/act/ckpt/pick_cup_cropped
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






