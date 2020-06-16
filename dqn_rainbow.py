#!/usr/bin/env python3
import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common
import datetime
import glob
import matplotlib.pyplot as plt
import os

from sacred import Experiment
from sacred.observers import MongoObserver

from gym.envs.registration import register

from utils import calc_loss,load_ckp

register(
    id='SimpleHighway-v1',
    entry_point='env.simple_highway.simple_highway_env:SimpleHighway',
)



experiment_name = "driving_behavior"
sacred_ex = Experiment(experiment_name)
now = datetime.datetime.now()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    sacred_ex.observers.append(MongoObserver(url='localhost:27017',
                                  db_name='sacred'))
except ConnectionError :
    print("MongoDB instance should be running")
             

@sacred_ex.config
def dqn_cfg():
    seed = 123523
    num_lane = 3
    level_k = 1
    agent_level_k = level_k -1
    TRAIN = True
    LOAD_SAVED_MODEL = False
    MODEL_PATH_FINAL = "best_"+str(agent_level_k)
    SAVE_NAME = "level" + str(agent_level_k)
    RENDER = True
    w1 = 0.6
    w2 = 0.3
    w3 = 0.1
    w4 = 0.1


#=======================================================
# n-step
REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 200000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 500

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(input_shape[0], 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(input_shape[0], 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        #fx = x.float() / 256
        #conv_out = self.conv(fx).view(fx.size()[0], -1)
        batch_size = x.size()[0]
        fx = x.float()
        val_out = self.fc_val(fx.view(fx.size()[0], -1)).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(fx.view(fx.size()[0], -1)).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


current_path = os.path.dirname(os.path.abspath(__file__))
@sacred_ex.automain
def main(_config,_run):
    
    logger = _run
    SAVE_NAME = _config['SAVE_NAME']
    LOAD_SAVED_MODEL = _config['LOAD_SAVED_MODEL']
    MODEL_PATH_FINAL = _config['MODEL_PATH_FINAL']
    total_steps = 1000000

    params = common.HYPERPARAMS['gamePlay2']
    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(params['env_name'],glob_conf=_config,logger=logger)
    #env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow-beta200")
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    
    #net.load_state_dict(torch.load(  ))
    name_load = current_path +"/models" +MODEL_PATH_FINAL
    if _config['LOAD_SAVED_MODEL']:
        mdl, opt, lss = load_ckp(MODEL_PATH_FINAL, net, optimizer)
        net = mdl
        optimizer = opt

    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)
    # change the step_counts to change multi step prediction
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    today = datetime.datetime.now()
    todays_date_full = str(today.year) + "_" + str(today.month) + "_" + str(today.day) + "_"
    todays_date_full += str(today.hour) + "_" + str(today.minute) + "_" + str(today.second)
    folder_name = todays_date_full +"_"+experiment_name
    results_dir = current_path + "/results/" + folder_name
    results_dir_weights = results_dir + "/weights"
    os.makedirs(results_dir)
    os.makedirs(results_dir_weights)

    frame_idx = 0
    beta = BETA_START
    best_mean_reward = 0.0
    eval_states = None
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while frame_idx < total_steps:
            frame_idx += 1
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                # start saving the model after actual training begins
                if frame_idx > 100:
                    if best_mean_reward is None or best_mean_reward < reward_tracker.mean_reward:
                        torch.save(net.state_dict(),
                                   SAVE_NAME + "-best.dat")

                        if best_mean_reward is not None:
                            print("Best mean reward updated %.3f -> %.3f, model saved" % \
                                  (best_mean_reward, reward_tracker.mean_reward))
                        if not reward_tracker.mean_reward == 0:
                            best_mean_reward = reward_tracker.mean_reward

                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < params['replay_initial']:
                continue
            if eval_states is None:
                eval_states, _, _ = buffer.sample(STATES_TO_EVALUATE, beta)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'] ** REWARD_STEPS, device=device)

            # if frame_idx % 10000 == 0:
            if frame_idx % 5000 == 0:
                checkpoint = ({
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss_v,
                    'num_step': frame_idx
                })
                torch.save(checkpoint, results_dir_weights + "/rainbow" + str(frame_idx) + "step.dat")

                # Save network parameters as histogram
                for name, param in net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), frame_idx)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            if logger:
                loss_v.item()
                logger.log_scalar("loss", loss_v.item())