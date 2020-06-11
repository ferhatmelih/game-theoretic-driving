import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise

    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        # cnn DQN for like volvo paper:
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(4, 1), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4)  # number of vehicles
        )
        #conv_out_size = self._get_conv_out(input_shape)
        # nn.Linear(input_shape[0],256 ...
        self.fc = nn.Sequential(
            nn.Linear(34, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        #fx = x.float() / 256
        #conv_out = self.conv(fx).view(fx.size()[0], -1)
        batch_size = x.size()[0]
        # 18 5 vehicles with accel=> 4-4 (other veh) + 2 ego
        x = torch.tensor(x).reshape(batch_size, 1, 18, 1)
        indices_ego = torch.tensor([0, 1]).cuda()
        indices_o = torch.tensor(range(2, 18)).cuda()
        x_ego = torch.index_select(x, 2, indices_ego)
        xo = torch.index_select(x, 2, indices_o)

        conv_out = self.conv(xo.float())
        conv_out = conv_out.float().reshape(batch_size, 32)
        x_ego = x_ego.float().reshape(batch_size, 2)
        merged = torch.cat([x_ego, conv_out], dim=1)
        #fx = x.float()
        return self.fc(merged.view(merged.size()[0], -1))
