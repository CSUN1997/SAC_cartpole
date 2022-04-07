import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
torch.autograd.set_detect_anomaly(True)


def to_tensor(vec):
    if not torch.is_tensor(vec):
        vec = torch.FloatTensor(vec).cuda()
    if vec.dim() <= 1:
        vec = vec.reshape([1, -1])
    return vec


class Actor(nn.Module):
    def __init__(self, state_space, action_space):
        super(Actor, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
            nn.Softmax(dim=1)
        )

    def forward(self, state):
        return self.actor(state)


class Critic(nn.Module):
    def __init__(self, state_space, action_space):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_space + action_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state, action):
        return self.critic(torch.cat([state, action], dim=1))


class SAC(nn.Module):
    def __init__(self, state_space, action_space, lr_actor=0.0001, lr_critic=0.0005,
                 gamma=0.9, alpha=0.5, update_interval=200, EPS_START=0.5, EPS_END=0.05, EPS_DECAY=200):
        super(SAC, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.alpha = alpha
        self.update_interval = update_interval
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY

        self.actor = Actor(state_space, action_space).cuda()
        self.actor_target = Actor(state_space, action_space).cuda()

        self.critic1 = Critic(state_space, action_space).cuda()
        self.critic1_target = Critic(state_space, action_space).cuda()

        self.critic2 = Critic(state_space, action_space).cuda()
        self.critic2_target = Critic(state_space, action_space).cuda()

        self.update_target()

        # self.optimizer = optim.Adam([
        #     {'params': self.actor.parameters(), 'lr': lr_actor},
        #     {'params': self.critic1.parameters(), 'lr': lr_critic},
        #     {'params': self.critic2.parameters(), 'lr': lr_critic}
        # ])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr_actor)
        self.critic_optimizer1 = optim.Adam(self.critic1.parameters(), lr_critic)
        self.critic_optimizer2 = optim.Adam(self.critic2.parameters(), lr_critic)

        self.steps = 0

    def update_target(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

    def choose_action(self, state):
        action_probs = self.actor(to_tensor(state))
        # action_probs = torch.cat([action_probs, 1 - action_probs], dim=1)
        dist = Categorical(action_probs)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        np.exp(-1. * self.steps / self.EPS_DECAY)
        if np.random.rand() > eps_threshold:
            action_probs = self.actor(to_tensor(state))
            action = torch.argmax(action_probs).item()
        else:
            action = np.random.choice(self.action_space)
        self.steps += 1
        if self.steps % self.update_interval == 0:
            self.update_target()
        return action, action_probs.detach().cpu().numpy()

    def learn(self, s, a, r, sn, d):
        state = torch.cat([to_tensor(ss) for ss in s], dim=0)
        action = torch.cat([to_tensor(aa) for aa in a], dim=0)
        reward = to_tensor(r).squeeze()
        reward = (reward - torch.mean(reward)) / (torch.std(reward) + 1e-5)
        state_next = torch.cat([to_tensor(ss) for ss in sn], dim=0)
        done = to_tensor(d).squeeze()

        loss = 0.
        # update critic
        action_probs_next = self.actor_target(state_next)
        # action_probs_next = torch.cat([action_probs_next, 1 - action_probs_next], dim=1)
        dist = Categorical(action_probs_next)
        action_next = dist.sample()
        log_prob_next = dist.log_prob(action_next)
        # log_prob_next = torch.log(action_probs_next)
        Q_next = torch.minimum(self.critic1_target(state_next, action_probs_next),
                               self.critic2_target(state_next, action_probs_next))
        Q_target = reward - self.gamma * (1 - done) * (Q_next.squeeze() - self.alpha * log_prob_next)
        Q1 = self.critic1(state, action)
        Q2 = self.critic2(state, action)
        critic_loss = F.mse_loss(Q1.squeeze(), Q_target.detach()) + F.mse_loss(Q2.squeeze(), Q_target.detach())
        self.critic1.zero_grad()
        self.critic2.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic1.parameters(), 0.3)
        nn.utils.clip_grad_norm_(self.critic2.parameters(), 0.3)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()

        # update actor
        action_probs = self.actor(state)
        # action_probs = torch.cat([action_probs, 1 - action_probs], dim=1)
        dist = Categorical(action_probs)
        action_cur = dist.sample()
        Q1 = self.critic1(state, action_probs)
        Q2 = self.critic2(state, action_probs)
        min_Q = torch.minimum(Q1, Q2)
        actor_loss = -torch.mean(min_Q - self.alpha * dist.log_prob(action_cur))

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1)
        self.actor_optimizer.step()

        return actor_loss + critic_loss

