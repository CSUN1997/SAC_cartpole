from itertools import count

import numpy as np
import gym
from sac_model import SAC
from replay_buffer import ReplayBuffer
import pandas as pd


def play(batch_size, n_episodes):
    recorder = {
        'reward': [],
        'steps': []
    }
    for episode in range(n_episodes):
        epi_reward = 0.
        state = env.reset()
        step = 0
        while True:
            action, action_probs = sac.choose_action(state)
            # print(action, action_probs)
            state_next, reward, done, _ = env.step(action)
            memory.remember(state, action_probs, reward, state_next, done)
            if done or step >= 50:
                break
            if len(memory) > batch_size:
                batch = np.random.choice(len(memory), batch_size)
                s, a, r, sn, d = memory.sample(batch)
                sac.learn(s, a, r, sn, d)
            state = state_next
            epi_reward += reward
            step += 1
        if episode % 10 == 0:
            print(f'episode: {episode}, reward: {epi_reward}')
        recorder['steps'].append(step)
        recorder['reward'].append(epi_reward)
    return recorder


if __name__ == '__main__':
    env = gym.make('CartPole-v0').unwrapped
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(alpha)
        memory = ReplayBuffer(capacity=100000)
        sac = SAC(env.observation_space.shape[0], env.action_space.n, lr_actor=0.001, lr_critic=0.003, alpha=alpha)
        recorder = play(batch_size=16, n_episodes=200)
        pd.DataFrame(recorder).to_csv(f'./result_alpha_{alpha}.csv', index=False)

