import sys
sys.path.append('.')
"""
This is an example on how to use the two player Wimblepong environment
with two SimpleAIs playing against each other
"""
import matplotlib.pyplot as plt
from random import randint
import pickle
import gym
import numpy as np
import torch
import argparse
import wimblepong
from PIL import Image
# from PPO_pong import PPO_Agent
from PPO_pong_rewards_normalized import NNPolicy, image_to_grey, PPO_Agent

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--load_dir", default="pingfuckingpong/", type=str, help="Path to look for the model")
parser.add_argument("--n_episodes", default=100, type=int, help="Time to train the agent for")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualMultiplayer-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
# Number of episodes/games to play
episodes = 100000

# Define the player IDs for both SimpleAI agents
player_id = 1
opponent_id = 3 - player_id
opponent = wimblepong.SimpleAi(env, opponent_id)
mem_size = 256
policy = NNPolicy(1, 256, env.action_space.n)
policy.try_load(args.load_dir)
player = PPO_Agent(policy)


# policy2 = NNPolicy(1, 256, env.action_space.n)
# policy2.try_load(args.load_dir)
# player2 = PPO_Agent(policy)

# Set the names for both SimpleAIs
env.set_names(player.get_name(), opponent.get_name())

win1 = 0
for i in range(0,episodes):
    hidden = torch.zeros(1, mem_size)
    hidden2 = torch.zeros(1, mem_size)
    done = False
    (ob1, ob2) = env.reset()
    while not done:
        # Get the actions from both SimpleAIs
        action1, hidden = player.select_action((image_to_grey(ob1), hidden.detach()))
        # action2, hidden2 = player.select_action((image_to_grey(ob2), hidden2.detach()))
        action2 = opponent.get_action()
        # Step the environment and get the rewards and new observations
        (ob1, ob2), (rew1, rew2), done, info = env.step((action1, action2))
        #img = Image.fromarray(ob1)
        #img.save("ob1.png")
        #img = Image.fromarray(ob2)
        #img.save("ob2.png")
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
        if done:
            observation= env.reset()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
