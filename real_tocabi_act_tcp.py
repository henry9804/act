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
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32
from tocabi_msgs.msg._positionCommand import positionCommand
# TCP image
import socket
import struct
import multiprocessing as mp
from multiprocessing import Manager
import cv2


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
ready_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
              0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
              0.0, 0.0, 0.0, 
              0.0, -0.3, 1.571, -1.2, -1.571, 1.5, 0.4, -0.2, 
              0.0, 0.3, 
              0.0, 0.3, -1.571, 1.2, 1.571, -1.5, -0.4, 0.2]

hand_joint_name  = ["aa1" , "aa2" , "aa3" , "aa4" ,
                    "act1", "act2", "act3", "act4",
                    "dip1", "dip2", "dip3", "dip4",
                    "mcp1", "mcp2", "mcp3", "mcp4",
                    "pip1", "pip2", "pip3", "pip4"]
hand_open_state = np.array([0.0   , 0.0   , 0.0   , 0.0   ,
                            0.0550, 0.0550, 0.0550, 0.0550,
                            0.1517, 0.1517, 0.1517, 0.1517,
                            0.0687, 0.0687, 0.0687, 0.0687,
                            0.0088, 0.0088, 0.0088, 0.0088])
hand_close_state= np.array([0.6   , 0.0   , 0.0   , 0.0   ,
                            0.6645, 0.6645, 0.6645, 0.6645,
                            0.8379, 0.8379, 0.8379, 0.8379,
                            0.8306, 0.8306, 0.8306, 0.8306,
                            0.9573, 0.9573, 0.9573, 0.9573])

joint_index = [12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
FPS = 30
log_dir = '/home/dyros/act/logs'

def receive_image(sock):
    # Receive the size of the image
    data = b""
    while len(data) < 4:
        packet = sock.recv(4 - len(data))
        if not packet:
            return None
        data += packet
    size = struct.unpack(">L", data)[0]
    
    # Receive the image data based on the received size
    data = b""
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data += packet
    
    return data

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
        self.max_timesteps = int(episode_len * 10)

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

        self.q_current = np.zeros(33)
        self.q_init = np.zeros(33)
        self.hand_state = 0.0   # change this value appropriately ex. pick->0.0 place->1.0

        self.img_buf = None
        
        self.position_command_pub = rospy.Publisher("/tocabi/positioncommand", positionCommand, queue_size=1)
        self.hand_open_pub = rospy.Publisher('/tocabi_hand/on', Bool, queue_size=1)
        self.current_state_pub = rospy.Publisher('/current_state', DisplayRobotState, queue_size=1)
        self.goal_state_pub = rospy.Publisher('/goal_state', DisplayRobotState, queue_size=1)

    def to_ready_pose(self):
        pc_msg = positionCommand()
        pc_msg.position = ready_pose
        pc_msg.traj_time = 4
        pc_msg.gravity = True
        pc_msg.relative = False
        self.position_command_pub.publish(pc_msg)
        time.sleep(5)

    def joint_callback(self, joint_msg):
        self.q_current = np.array(joint_msg.position)
        current_state = DisplayRobotState()
        current_state.state.joint_state.header.stamp = rospy.Time.now()
        current_state.state.joint_state.name = [*joint_name, *hand_joint_name]
        if self.hand_state < 0.5:
            current_state.state.joint_state.position = np.concatenate([self.q_current, hand_open_state])
        else:
            current_state.state.joint_state.position = np.concatenate([self.q_current, hand_close_state])
        self.current_state_pub.publish(current_state)

    def mode_callback(self, msg):
        if msg.data == 0:
            self.to_ready_pose()
        elif msg.data == 1:
            if not self.start:
                print('start!')
                self.t = 0
                self.q_init = self.q_current.copy()
                self.start = True
            else:
                print('[WARNING] ACT inference is already in progress!')
        elif msg.data == 2:
            if self.start:
                print('end!')
                self.start = False
                log = self.all_time_actions.clone().detach()
                np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S.npy', time.localtime())), log.cpu().numpy())
            else:
                print('[WARNING] ACT inference has been already terminated!')
        else:
            print('[WARNING] wrong mode input!\n',
                  '          0: move to reay pose\n'
                  '          1: start ACT inferencing\n'
                  '          2: end ACT inferencing')

    def inference(self):
        with torch.inference_mode():
            start = time.time()

            qpos = [self.q_current[idx] for idx in joint_index]
            qpos.append(self.hand_state)
            qpos_numpy = np.array(qpos)
            qpos = self.pre_process(qpos_numpy)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            
            images = {}
            for cam_name in self.camera_names:
                # Decode the image
                cv_image = cv2.imdecode(np.frombuffer(self.img_buf[cam_name], dtype=np.uint8), cv2.IMREAD_COLOR)

                # process the image
                if cam_name == 'right':
                    cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
                cropped_image = cv_image[480:,480:1440]
                resized_image = cv2.resize(cropped_image, (640,480), interpolation=cv2.INTER_AREA)
                rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

                # save to image instance
                normalized_image = torch.from_numpy(rgb_image / 255.0).float()
                images[cam_name] = normalized_image
            curr_image = torch.stack([rearrange(images[cam_name], 'h w c -> c h w') for cam_name in self.camera_names], axis=0)
            curr_image = curr_image.cuda().unsqueeze(0)

            ### query policy
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)
            inference_time = time.time() - start
            if self.t == 0:
                next_step = 0
            else:
                next_step = int(inference_time * FPS)
            if self.temporal_agg:
                self.all_time_actions[[self.t], self.t:self.t+self.num_queries] = self.all_actions
                self.t += next_step
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                self.t += next_step
                raw_action = self.all_actions[:, self.t % self.query_frequency]

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = self.post_process(raw_action)
            target_qpos = action[:-1]
            target_hand = action[-1]

            ### step the environment
            print(self.t, ': ', action)
            print('  inference  time: ', inference_time*1000, 'ms')

            # visualize last action chunks
            raw_final_action = self.all_actions[:, -1]
            final_action = self.post_process(raw_final_action.squeeze(0).cpu().numpy())
            goal_state = DisplayRobotState()
            goal_state.state.joint_state.header.stamp = rospy.Time.now()
            goal_state.state.joint_state.name = [*joint_name, *hand_joint_name]
            if final_action[-1] < 0.5:
                goal_state.state.joint_state.position = np.concatenate([self.q_init, hand_open_state])
            else:
                goal_state.state.joint_state.position = np.concatenate([self.q_init, hand_close_state])
            goal_state.state.joint_state.position[joint_index] = final_action[:-1]
            self.goal_state_pub.publish(goal_state)

            # tocabi control
            pc_msg = positionCommand()
            pc_msg.position = self.q_init.copy()
            for j, q in zip(joint_index, target_qpos):
                if j == 13: # for tocabi waist safety limit
                    pc_msg.position[j] = min(q, 0.55)
                else:
                    pc_msg.position[j] = q
            # rasie left arm as bending waist
            pc_msg.position[19] -= pc_msg.position[13]
            post_processs_time = time.time() - start
            next_step = int(post_processs_time * FPS)
            print('post process time: ', (post_processs_time - inference_time)*1000, 'ms')
            if self.t == 0:
                pc_msg.traj_time = 1/FPS
            else:
                pc_msg.traj_time = (1+next_step)/FPS - post_processs_time
            pc_msg.gravity = True
            pc_msg.relative = False
            self.position_command_pub.publish(pc_msg)

            # hand control
            if self.hand_state < 0.5 and target_hand >= 0.5:
                # grasp
                hand_msg = Bool()
                hand_msg.data = True
                self.hand_open_pub.publish(hand_msg)
                self.hand_state = 1.0
            elif self.hand_state >= 0.5 and target_hand < 0.5:
                # stretch
                hand_msg = Bool()
                hand_msg.data = False
                self.hand_open_pub.publish(hand_msg)
                self.hand_state = 0.0

            self.t += 1

            total_time = time.time() - start
            print('    total    time: ', total_time*1000, 'ms')
            if self.t == 0:
                time.sleep(1/FPS)
            else:
                time.sleep((1+next_step)/FPS - post_processs_time)

        return total_time

def visualize_img(camera_names, img_buf, stop_event):
    plt.ion()
    fig, axs = plt.subplots(1, len(camera_names), squeeze=False)
    imshow = {}
    for ax, cam_name in zip(axs[0], camera_names):
        imshow[cam_name] = ax.imshow(np.zeros([480, 640, 3]))
        ax.set_title(cam_name)
        ax.axis('off')
        
    while not stop_event.is_set():
        for cam_name in camera_names:
            if img_buf[cam_name] is None:
                continue
            # Decode the image
            cv_image = cv2.imdecode(np.frombuffer(img_buf[cam_name], dtype=np.uint8), cv2.IMREAD_COLOR)
            if cv_image is None:
                continue

            # process the image
            if cam_name == 'right':
                cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
            cropped_image = cv_image[480:,480:1440]
            resized_image = cv2.resize(cropped_image, (640,480), interpolation=cv2.INTER_AREA)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            imshow[cam_name].set_data(rgb_image)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.01)

    plt.close()

def run_server(cam_name, img_buf, stop_event):
    host = "10.112.1.187"  # change the host to ip address of current device
    cam_name_to_port = {'left':41117,'right':41119}
    port = cam_name_to_port[cam_name]

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on port {port}")
    
    client_socket, addr = server_socket.accept()
    print(f"Connection from {addr}")

    befo_time = time.time()
    dts = np.zeros(FPS)
    idx = 0

    while not stop_event.is_set():
        try:
            dt = time.time() - befo_time
            dts[idx] = dt
            idx += 1

            if idx == FPS:
                idx = 0
                print(f'[{cam_name}] frame rate:', 1/dts.mean())
            
            befo_time = time.time()
            
            image_data = receive_image(client_socket)
            if image_data is None:
                print('no image received')
                break
            img_buf[cam_name] = image_data

        except Exception as e:
            print(f"Error: {e}")
            break
    
    client_socket.close()
    server_socket.close()
    print(f'end {cam_name} receiver.')

def main(args):
    tocabi_act = TocabiAct(vars(args))

    with Manager() as manager:
        tocabi_act.img_buf = manager.dict()
        for cam_name in tocabi_act.camera_names:
            tocabi_act.img_buf[cam_name] = None
        stop_event = mp.Event()
        processes = []

        # image receiving server
        for cam_name in tocabi_act.camera_names:
            p = mp.Process(target=run_server, args=(cam_name, tocabi_act.img_buf, stop_event))
            p.start()
            processes.append(p)

        # visualizing images
        p = mp.Process(target=visualize_img, args=(tocabi_act.camera_names, tocabi_act.img_buf, stop_event))
        p.start()
        processes.append(p)

        rospy.init_node('tocabi_act', anonymous=True)

        joint_sub = rospy.Subscriber("/tocabi/jointstates", JointState, tocabi_act.joint_callback)

        mode_sub = rospy.Subscriber("/tocabi/act/mode", Int32, tocabi_act.mode_callback)

        inf_time = []

        while not rospy.is_shutdown():
            if tocabi_act.start:
                total_time = tocabi_act.inference()
                inf_time.append(total_time)
                if tocabi_act.t == tocabi_act.max_timesteps:
                    print('end!')
                    tocabi_act.start = False
                    log = tocabi_act.all_time_actions.clone().detach()
                    np.save(os.path.join(log_dir, time.strftime('%Y%m%d_%H%M%S.npy', time.localtime())), log.cpu().numpy())

        print('gracefully shutdown ...')
        stop_event.set()
        for p in processes:
            p.join()

        plt.title(f'inference time\nmin: {min(inf_time[1:]):.4f} avg: {np.mean(inf_time[1:]):.4f} max: {max(inf_time[1:]):.4f}')
        plt.plot(inf_time[1:])
        plt.savefig(os.path.join(log_dir, time.strftime('inference_time_%Y%m%d_%H%M%S.png', time.localtime())))
        plt.close()

'''
run with following args
python real_tocabi_act_tcp.py \
--policy_class ACT --kl_weight 10 --hidden_dim 512 --batch_size 8 \
--dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 0 --temporal_agg \
--chunk_size 30 --task_name real_tocabi_open --ckpt_dir /media/dyros/SSD2TB/act/ckpt/1_open/stereo_smoothed
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
