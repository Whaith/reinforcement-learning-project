"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
from random import randint
from collections import deque
import pickle
import gym
import numpy as np
import torch
import argparse
import cv2
from PIL import Image
from tensorboardX import SummaryWriter

import wimblepong
from prototypes.PPO_agent import PPO_Agent, CNN_policy


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

def image_to_grey(obs, target_reso=(84, 84)):
    return np.dot(cv2.resize(obs[...,:3], dsize=target_reso), [0.2989, 0.5870, 0.1140]).astype('float32')/255.0

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
N_ACTIONS = env.action_space.n
N_STACKED_FRAMES = 3
IMAGE_DIMS = (84, 84)
# Number of episodes/games to play
episodes = 10000000
# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
player = PPO_Agent(policy=CNN_policy, policy_kwargs={'num_actions': N_ACTIONS})
player.load_model()
# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())
# PPO SETTINGS
device = player.policy.get_device()

observations_queue = deque([], maxlen=3)
# method used in training
def reset_deque(queue, n_elems_to_reset):
    for i in range(n_elems_to_reset):
        queue.append(np.zeros(IMAGE_DIMS).astype('float32'))
reset_deque(observations_queue, N_STACKED_FRAMES)
win1 = 0
T = 0
initially_updated = False
for i in range(0,episodes):
    done = False
    # reset observations to agents
    observation = env.reset()
    # observation to agent 1
    # set first 2 elements to 0
    reset_deque(observations_queue, 2)
    # set last element to actual observation
    observations_queue.append(image_to_grey(observation[0]))
    total_r = 0
    writer = SummaryWriter()
    while not done:
        # Get the actions from both SimpleAIs
        observation_1 = np.stack([observations_queue])
        action1 = player.select_action(observation_1)
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        player.policy.rewards.append(rew1)
        total_r += rew1
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        observations_queue.append(image_to_grey(ob1))
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            if (i+1) % 100 == 0:
                player.save_models()
            player.train_on_rollout(writer, T)
            if initially_updated:
                player.set_old_policy_to_new()
            initially_updated = True
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
        T += 1
    writer.add_scalar("Ep reward", total_r, i)

