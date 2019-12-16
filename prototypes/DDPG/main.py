# import sys
# sys.path.append('.')
import gym
import numpy as np
import argparse
from collections import deque
from cp_cont import CartPoleEnv
from DDPG_agent import DDPG_Agent
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="ContinuousCartPole-v0", type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=400, type=int, help='N_episodes')
    parser.add_argument('--T', default=300, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render", action="store_true", help="Render the environment mode")
    parser.add_argument("--lograte", default=100, type=int, help="Log frequency")
    # parser.add_argument("--render", default=, help="Render the environment mode")
    return parser.parse_args()


def learn_episodic_DDPG(args):
    env = gym.make('Pendulum-v0')
    ob_sp = env.observation_space.shape[0]
    act_sp = env.action_space.shape[0]

    writer = SummaryWriter()
    running_rewards = deque([], maxlen=args.lograte)
    agent = DDPG_Agent(ob_sp, act_sp, -2, 2, writer)
    for ep in range(args.n_eps):
        observation = env.reset()
        done = False
        epr = 0
        for t in range(args.T):
            action = agent.get_action(observation)
            next_obs, reward, done, _ = env.step([action])
            agent.store_transition(observation, action, reward, next_obs, done)
            agent.train()
            epr += reward
            observation = next_obs
            
            if args.render:
                env.render()

            if done:
                break
        writer.add_scalar('Epr', epr, ep)
        running_rewards.append(epr)
        if (ep + 1) % args.lograte == 0:
            print(f"episode: {ep}, running episode rewards: {np.mean(running_rewards)}")
        # TODO ADD logging to the
        
    return 0


if __name__ == '__main__':
    N_EPS = 10000
    args = get_args()
    print(f"initializing training with args: {args}")
    # rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
    rewards_DDPG = learn_episodic_DDPG(args)
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legend()
    # plt.show()
