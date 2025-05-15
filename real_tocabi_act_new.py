from geometry_msgs.msg import PoseStamped, PoseArray, PolygonStamped
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, Int32, String
from scipy.spatial.transform import Rotation
from roma.mappings import special_gramschmidt
import numpy as np
import rospy
import os
import time
import cv2
import matplotlib.pyplot as plt

import argparse
import pickle
import torch
import torch.nn.functional as F
from policy import ACTPolicy, ACTTaskPolicy
from constants import SIM_TASK_CONFIGS
from transform import TF_mat

log_dir = '/home/lyh/projects/embodied_ai/act/logs'
FPS = 15
JOINT_NAMES = ["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint",
               "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
               "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint",
               "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint",
               "Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
               "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Armlink_Joint",
               "L_Elbow_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
               "Neck_Joint", "Head_Joint",
               "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Armlink_Joint",
               "R_Elbow_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"]
STATE_NAMES = ["R_11", "R_21", "R_31", 
               "R_12", "R_22", "R_32", 
               "R_13", "R_23", "R_33",
               "X", "Y", "Z"]
joint_index = [25, 26, 27, 28, 29, 30, 31, 32]

def im_msg_2_cv_img(img_msg, rotate = False, dir=None):
    image_array = np.frombuffer(img_msg.data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if dir != None:
        stamp = img_msg.header.stamp
        filename = f'{img_msg.header.seq}_{stamp.secs}_{stamp.nsecs:09}.{img_msg.format}'
        cv2.imwrite(os.path.join(dir, filename), image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return rgb_image
    # cropped_image = rgb_image[:,160:1760]
    # resized_image = cv2.resize(cropped_image, (640,480), interpolation=cv2.INTER_AREA)
    # if rotate:
    #     resized_image = cv2.rotate(resized_image, cv2.ROTATE_180)

    # return resized_image

class TocabiAct:
    def __init__(self, args):
        ckpt_path = args['ckpt_dir']
        self.policy_class = args['policy_class']
        task_name = args['task_name']
        self.temporal_agg = args['temporal_agg']

        task_config = SIM_TASK_CONFIGS[task_name]
        episode_len = task_config['episode_len']
        self.camera_names = task_config['camera_names']
        self.max_timesteps = int(episode_len * 1.5)
        self.hand_open = False
        self.pelvis_TF = TF_mat()
        self.head_TF = TF_mat()

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
                        }

        if self.policy_class == 'ACT':
            joint_sub = rospy.Subscriber("/tocabi/jointstates", JointState, self.joint_callback)
            self.joint_target_pub = rospy.Publisher("/tocabi/act/joint_target", JointState, queue_size=1)
            self.state_dim = task_config['model_dof']
            policy_config['state_dim'] = self.state_dim
            self.policy = ACTPolicy(policy_config)
        elif self.policy_class == 'ACTTask':
            robot_poses_sub = rospy.Subscriber("/tocabi/robot_poses", PoseArray, self.robot_poses_callback)
            self.pose_target_pub = rospy.Publisher("/tocabi/act/pose_target", PoseStamped, queue_size=1)
            self.state_dim = 10
            policy_config['state_dim'] = self.state_dim
            self.policy = ACTTaskPolicy(policy_config)
        else:
            raise ValueError(f'Unknown policy class: {self.policy_class}')
        
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
        print(loading_status)
        self.policy.cuda()
        self.policy.eval()
        print(f'Loaded: {ckpt_path}')

        ckpt_dir = os.path.dirname(ckpt_path)
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        self.pre_process = lambda s_qpos: (s_qpos - self.stats['qpos_mean']) / self.stats['qpos_std']

        if self.temporal_agg:
            self.query_frequency = 1
            self.num_queries = policy_config['num_queries']
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
        else:
            self.query_frequency = policy_config['num_queries']

        self.t = 0
        self.terminate = True
        self.all_actions = None
        self.img_msgs = {}

        self.hand_open_pub = rospy.Publisher("/tocabi_hand/on", Bool, queue_size=1)

        self.img_log = []
        self.state_log = []
        self.action_log = []
        self.inf_time_log = []

    def post_process_task(self, raw_action):
        raw_action = raw_action.squeeze()
        # unnormalize position action
        pos = raw_action[6:9].cpu().numpy()
        pos = pos * self.stats['action_std'][9:12] + self.stats['action_mean'][9:12]
        # 6D GSO representation to quternion
        R = special_gramschmidt(raw_action[:6].reshape(2, 3).transpose(0,1))
        R = R.cpu().numpy()
        quat = Rotation.from_matrix(R).as_quat()
        # apply sigmoid to hand state action
        hand = F.sigmoid(raw_action[9])
        hand = hand.cpu().numpy()

        self.action_log.append([*R.transpose().flatten(), *pos, hand])

        return pos, quat, hand

    def post_process_joint(self, raw_action):
        raw_action = raw_action.squeeze()
        # unnormalize joint action
        joint = raw_action[:-1].cpu().numpy()
        joint = joint * self.stats['action_std'][:-1] + self.stats['action_mean'][:-1]
        # apply sigmoid to hand state action
        hand = F.sigmoid(raw_action[-1])
        hand = hand.cpu().numpy()

        self.action_log.append([*joint, hand])

        return joint, hand
    
    def save_plots(self, plot_dir):
        # plot inference time
        inf_time = np.array(self.inf_time_log[1:])
        plt.plot(inf_time)
        plt.title(f'Inference Time\nmin: {inf_time.min():.4f}, avg: {inf_time.mean():.4f}, max: {inf_time.max():.4f}')
        plt.savefig(os.path.join(plot_dir, 'inference_time.svg'))
        plt.close()
        # plot state and action
        state = np.array(self.state_log)
        action = np.array(self.action_log)
        state_dim = state.shape[1]
        fig, axes = plt.subplots(state_dim, 1, figsize=(state_dim, 2 * state_dim))
        for i in range(state_dim-1):
            axes[i].plot(state[:, i])
            axes[i].plot(action[:, i])
            if self.policy_class == 'ACT':
                axes[i].set_title(JOINT_NAMES[joint_index[i]])
            elif self.policy_class == 'ACTTask':
                axes[i].set_title(STATE_NAMES[i])
        axes[-1].plot(state[:, -1], label='state')
        axes[-1].plot(action[:, -1], label='action')
        axes[-1].legend()
        axes[-1].set_title('Hand_State')
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'state_action.svg'))
        plt.close()

    def robot_poses_callback(self, robot_poses_msg):
        self.pelvis_TF = TF_mat.from_pose_msg(robot_poses_msg.poses[0])
        self.head_TF = TF_mat.from_pose_msg(robot_poses_msg.poses[2])
        rhand_TF = TF_mat.from_pose_msg(robot_poses_msg.poses[3])
        ##########################################################
        rhand_TF = TF_mat.mul(self.pelvis_TF.inverse(), rhand_TF)   # rhand_TF in pelvis frame
        # rhand_TF = TF_mat.mul(self.head_TF.inverse(), rhand_TF)     # rhand_TF in head frame
        ##########################################################
        rhand_vec = rhand_TF.as_matrix()[:3,:].transpose().flatten()
        self.state = np.concatenate([rhand_vec, [self.hand_open]])

    def joint_callback(self, joint_msg):
        joint_state = np.array([joint_msg.position[j] for j in joint_index])
        self.state = np.concatenate([joint_state, [self.hand_open]])
    
    def img_callback(self, img_msg, cam_name):
        self.img_msgs[cam_name] = img_msg

    def publish_action_msg(self):
        if not self.terminate:
            start = time.time()
            with torch.inference_mode():
                self.state_log.append(self.state)
                state = self.pre_process(self.state)
                state = torch.from_numpy(state).float().cuda().unsqueeze(0) # double -> float
                all_cam_images = []
                for cam_name in self.camera_names:
                    if cam_name.endswith('stereo'):
                        left_img = im_msg_2_cv_img(self.img_msgs[f'{cam_name[:-6]}left'])
                        right_img = im_msg_2_cv_img(self.img_msgs[f'{cam_name[:-6]}right'], rotate=True)
                        stereo_img = np.concatenate([left_img, right_img], axis=1) # width dimension
                        all_cam_images.append(stereo_img)
                    else:
                        img = im_msg_2_cv_img(self.img_msgs[cam_name])
                        all_cam_images.append(img)
                self.img_log.append(np.concatenate(all_cam_images, axis=0)) # height dimension
                all_cam_images = np.stack(all_cam_images, axis=0)
                curr_image = torch.from_numpy(all_cam_images / 255.0).float().cuda().unsqueeze(0)
                curr_image = torch.einsum('b k h w c -> b k c h w', curr_image)

                ### query policy
                if self.t % self.query_frequency == 0:
                    self.all_actions = self.policy(state, curr_image)
                if self.temporal_agg:
                    self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                    actions_for_curr_step = self.all_time_actions[:, self.t+24]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.1
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = self.all_actions[:, self.t % self.query_frequency]

                if self.policy_class == 'ACT':
                    ### post-process actions
                    target_joint, target_hand = self.post_process_joint(raw_action)
                    ### step the environment
                    print(self.t, ': ', target_joint, target_hand)
                    joint_target = JointState()
                    joint_target.header.stamp = rospy.Time.now()
                    joint_target.name = [JOINT_NAMES[j] for j in joint_index]
                    joint_target.position = target_joint
                    self.joint_target_pub.publish(joint_target)

                elif self.policy_class == 'ACTTask':
                    ### post-process actions
                    target_pos, target_quat, target_hand = self.post_process_task(raw_action)
                    ### step the environment
                    print(self.t, ': ', target_pos, target_quat, target_hand)
                    rhand_action_TF = TF_mat.from_vectors(target_pos, target_quat)
                    ########################################################
                    rhand_action_TF = TF_mat.mul(self.pelvis_TF, rhand_action_TF)   # rhand_action_TF pelvis frame -> world frame
                    # rhand_action_TF = TF_mat.mul(self.head_TF, rhand_action_TF)     # rhand_action_TF head frame -> world frame
                    ########################################################
                    pose_target = PoseStamped()
                    pose_target.header.frame_id = 'world'
                    pose_target.header.stamp = rospy.Time.now()
                    pose_target.pose = rhand_action_TF.as_pose_msg()
                    self.pose_target_pub.publish(pose_target)

                target_hand = target_hand > 0.5
                if self.hand_open != target_hand:
                    self.hand_open = target_hand
                    hand_open_msg = Bool()
                    hand_open_msg.data = target_hand
                    self.hand_open_pub.publish(hand_open_msg)
                
                self.t += 1
                if self.t >= self.max_timesteps:
                    self.terminate = True
            end = time.time()
            self.inf_time_log.append(end - start)
            return end - start
        else:
            return 0

    def terminate_callback(self, msg):
        if msg.data:
            self.terminate = True
            # wait for the current inference to finish
            time.sleep(0.1)
            # save logs
            log_name = time.strftime('%Y%m%d_%H%M%S')
            os.makedirs(os.path.join(log_dir, log_name), exist_ok=True)
            video_path = os.path.join(log_dir, log_name, 'video.mp4')
            # img input
            h, w, _ = self.img_log[0].shape
            out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
            for img in self.img_log:
                img = img[:, :, [2, 1, 0]]
                out.write(img)
            out.release()
            # state input and action output
            log_path = os.path.join(log_dir, log_name, 'log.pkl')
            with open(log_path, 'wb') as f:
                pickle.dump({'state_log': self.state_log,
                             'action_log': self.action_log,
                             'inf_time_log': self.inf_time_log}, f)
            self.save_plots(os.path.join(log_dir, log_name))
            print('Logs saved')
        else:
            # initialize variables
            self.t = 0
            self.all_time_actions = torch.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim]).cuda()
            self.img_log = []
            self.state_log = []
            self.action_log = []
            self.inf_time_log = []
            self.hand_open = False
            self.terminate = False

def main(args):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    rospy.init_node('tocabi_act', anonymous=True)

    tocabi_act = TocabiAct(vars(args))

    img_l_sub = rospy.Subscriber("/cam_LEFT/image_raw/compressed", CompressedImage, tocabi_act.img_callback, callback_args='left', queue_size=1)
    img_r_sub = rospy.Subscriber("/cam_RIGHT/image_raw/compressed", CompressedImage, tocabi_act.img_callback, callback_args='right', queue_size=1)
    terminate_sub = rospy.Subscriber("/tocabi/act/terminate", Bool, tocabi_act.terminate_callback)

    while not rospy.is_shutdown():
        inf_time = tocabi_act.publish_action_msg()
        if (inf_time < 1/FPS):
            time.sleep(1/FPS - (inf_time))
        else:
            time.sleep(1/FPS)


'''
run with following args
python real_tocabi_act_new.py \
--seed 0 --num_epochs 5000 --temporal_agg --task_name real_tocabi_pick_n_place \
--ckpt_dir /external/act/ckpt/real_tocabi_pick_n_place/ee_global/policy_val_best.ckpt --policy_class ACTTask
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=False, default=1)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=False, default=1e-5)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False, default=30)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False, default=3200)
    parser.add_argument('--temporal_agg', action='store_true')

    args = parser.parse_args()
    main(args)
