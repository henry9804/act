import os
import argparse
import numpy as np
import time
from matplotlib import pyplot as plt
# ACT
import pickle
import torch
from einops import rearrange
from policy import ACTPolicy
from constants import SIM_TASK_CONFIGS
# ROS
import rospy
from moveit_msgs.msg import DisplayRobotState
from sensor_msgs.msg import JointState, CompressedImage
from std_msgs.msg import Bool, Int32
# image
import cv2


joint_name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", 
              "panda_joint6", "panda_joint7", "panda_finger_joint1", "panda_finger_joint2"]
mode_name = ['default', 'home', 'teleoperate', 'ros_sub']

FPS = 10
log_dir = '/home/panda-pc/lyh/act/logs'

def im_msg_2_torch_img(img_msg, rotate=False, crop=False):
    image_array = np.frombuffer(img_msg.data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if rotate:
        image = cv2.rotate(image, cv2.ROTATE_180)
    if crop:
        image = image[:,160:1020]
        image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    normalized_image = torch.from_numpy(rgb_image / 255.0).float()

    return normalized_image

class PandaAct:
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

        ckpt_name = 'policy_epoch_76391_seed_0.ckpt'
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

        self.start = False
        self.t = 0
        self.all_actions = None

        self.q_current = np.zeros(8)
        self.im_msgs = {}
        self.images = {}
        for cam_name in self.camera_names:
            self.im_msgs[cam_name] = None
            self.images[cam_name] = torch.zeros([480, 640, 3])
        
        self.position_command_pub = rospy.Publisher("/panda/positioncommand", JointState, queue_size=1)
        self.goal_state_pub = rospy.Publisher('/goal_state', DisplayRobotState, queue_size=1)

        self.input_log = []
        self.output_log = []

    def img_callback(self, img_msg, cam_name):
        self.im_msgs[cam_name] = img_msg

    def joint_callback(self, joint_msg):
        self.q_current = np.array([*joint_msg.position[:7], 2*joint_msg.position[-1]])

    def mode_callback(self, msg):
        if msg.data == 3:
            if not self.start:
                print('start!')
                self.t = 0
                self.start = True
            else:
                print('[WARNING] ACT inference is already in progress!')
        else:
            if self.start:
                print('end!')
                self.start = False
                log = self.all_time_actions.clone().detach()
                np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S.npy', time.localtime())), log.cpu().numpy())
                np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S_input.npy', time.localtime())), np.array(self.input_log))
                np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S_output.npy', time.localtime())), np.array(self.output_log))
            else:
                print('[WARNING] ACT inference has been already terminated!')
            try:
                print(mode_name[msg.data], ' selected')
            except:
                print('[ERROR] Unindentified control mode!!!')

    def inference(self):
        with torch.inference_mode():
            start = time.time()

            qpos_numpy = np.array(self.q_current)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            for cam_name in self.camera_names:
                if cam_name == 'hand':
                    self.images[cam_name] = im_msg_2_torch_img(self.im_msgs[cam_name], rotate=True)
                else:
                    self.images[cam_name] = im_msg_2_torch_img(self.im_msgs[cam_name], crop=True)
            curr_image = torch.stack([rearrange(self.images[cam_name], 'h w c -> c h w') for cam_name in self.camera_names], axis=0)
            curr_image = curr_image.cuda().unsqueeze(0)
            img_process_time = time.time()-start

            ### query policy
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            inference_time = time.time() - start
            if self.temporal_agg:
                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t+4]
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
            self.input_log.append(qpos_numpy)
            self.output_log.append(action)
            print(' img process time: ', img_process_time*1000, 'ms')
            print('  inference  time: ', inference_time*1000, 'ms')

            # visualize last action chunks
            raw_final_action = self.all_actions[:, -1]
            final_action = self.post_process(raw_final_action.squeeze(0).cpu().numpy())
            goal_state = DisplayRobotState()
            goal_state.state.joint_state.header.stamp = rospy.Time.now()
            goal_state.state.joint_state.name = joint_name
            goal_state.state.joint_state.position = [*final_action[:-1], final_action[-1]/2, final_action[-1]/2]
            self.goal_state_pub.publish(goal_state)

            # panda control
            pc_msg = JointState()
            pc_msg.header.stamp = rospy.Time.now()
            pc_msg.name = joint_name[:-1]
            pc_msg.position = action

            post_processs_time = time.time() - start
            print('post process time: ', (post_processs_time - inference_time)*1000, 'ms')

            self.t += 1

            self.position_command_pub.publish(pc_msg)

            total_time = time.time() - start
            next_step = int(total_time * FPS)
            print('    total    time: ', total_time*1000, 'ms')
            if self.t == 0:
                time.sleep(1/FPS)
            else:
                time.sleep((1+next_step)/FPS - total_time)

        return total_time

def main(args):
    panda_act = PandaAct(vars(args))

    rospy.init_node('panda_act', anonymous=True)

    joint_sub = rospy.Subscriber("/joint_states", JointState, panda_act.joint_callback)
    left_img_sub = rospy.Subscriber("/left/rgb/image_raw/compressed", CompressedImage, panda_act.img_callback, callback_args='left')
    hand_img_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, panda_act.img_callback, callback_args='hand')
    right_img_sub = rospy.Subscriber("/right/rgb/image_raw/compressed", CompressedImage, panda_act.img_callback, callback_args='right')

    mode_sub = rospy.Subscriber("/franka_state_controller/control_mode", Int32, panda_act.mode_callback)

    inf_time = []

    while not rospy.is_shutdown():
        if panda_act.start:
            total_time = panda_act.inference()
            inf_time.append(total_time)
            if panda_act.t == panda_act.max_timesteps:
                print('end!')
                panda_act.start = False
                log = panda_act.all_time_actions.clone().detach()
                np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S.npy', time.localtime())), log.cpu().numpy())

    plt.title(f'inference time\nmin: {min(inf_time[1:]):.4f} avg: {np.mean(inf_time[1:]):.4f} max: {max(inf_time[1:]):.4f}')
    plt.plot(inf_time[1:])
    plt.savefig(os.path.join(log_dir, time.strftime('inference_time_%Y%m%d_%H%M%S.png', time.localtime())))
    plt.close()

'''
run with following args
python real_panda_act.py \
--policy_class ACT --kl_weight 10 --hidden_dim 512 --batch_size 8 \
--dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --temporal_agg \
--chunk_size 30 --task_name real_panda_pick_n_place --ckpt_dir ckpt/pick_n_place
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

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args = parser.parse_args()
    main(args)
