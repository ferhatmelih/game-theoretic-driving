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
now = datetime.datetime.now()

EVAL_ESTIMATOR = False ## this should be False to start training estimator.

MODEL_PATH_FINAL = "Rainbow30.dat"
SAVED_ESTIMATORS_FOLDER = "modelstobetested/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### FAIL ESTIM PARAMS ##############
ESTIMATOR_LR = 0.01
ESTIMATOR_BATCH = 8
ESTIMATOR_FEATURE_SIZE = 4
SAVE_MODELS_EACH = 300

TEST_FAIL_DETECTOR_EACH = 16


DELTA_TIME_FEATURE_VECTOR = 1.0

#=======================================================

EVAL_RUNS = 200

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

def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)
   

class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )

        #conv_out_size = self._get_conv_out(input_shape)
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

    # def forward(self, x):
    #     batch_size = x.size()[0]
    #     fx = x.float() / 256
    #     conv_out = self.conv(fx).view(batch_size, -1)
    #     val_out = self.fc_val(conv_out).view(batch_size, 1, N_ATOMS)
    #     adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
    #     adv_mean = adv_out.mean(dim=1, keepdim=True)
    #     return val_out + (adv_out - adv_mean)

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




if __name__ == "__main__":
    # I wamt to play pygame
    total_steps = 1000000

    params = common.HYPERPARAMS['gamePlay2']
    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()

    env = gym.make(params['env_name'])
    #env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-rainbow-beta200-2LaneStateBenchWoutAccel_sparsePlusSpeed")
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    if 'MODEL_PATH_FINAL' in locals():
        net.load_state_dict(torch.load(MODEL_PATH_FINAL))
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)
    # change the step_counts to change multi step prediction
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])


    if(EVAL_ESTIMATOR):
        pass
        #eval_saved_models(exp_source,estimator_linear=estimator_linear)
    else:
        frame_idx = 0
        beta = BETA_START
        best_mean_reward = 0.0
        eval_states = None
        finished_run_counter = -1
        batch_counter = 0
        prev_accident_num = 0
        with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
            while frame_idx < total_steps:
                frame_idx += 1
                buffer.populate(1)
                beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    # start saving the model after actual training begins
                    finished_run_counter+=1
                    total_accidents = exp_source.total_accidents
                    last_iteration_accident_num = total_accidents - prev_accident_num
                    if(last_iteration_accident_num>0):
                        pass
                        #print("accident occured frame: ",frame_idx)




