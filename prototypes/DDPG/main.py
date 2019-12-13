import gym
import numpy as np
from cp_cont import CartPoleEnv
from DDPG_agent import DDPG_Agent   

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default="ContinuousCartPole-v0", type=str, help='gym environment name')
    parser.add_argument('--n_eps', default=200, type=int, help='N_episodes')
    parser.add_argument('--T', default=300, type=int, help='maximum timesteps per episode')
    parser.add_argument("--render", action="store_true", help="Render the environment mode")
    parser.add_argument("--render", default, help="Render the environment mode")
    return parser.parse_args()


def learn_episodic_DDPG(args): 
    env = gym.make(args.env)
    ob_sp = env.observation_space.shape[0]
    act_sp = env.action_space.shape[0]

    agent = DDPG_Agent(ob_sp, act_sp, -np.inf, np.inf)
    
    for i in range()

# #     writer = SummaryWriter()
#     # env = gym.make('Pendulum-v0')
#     env = gym.make(env_name)
#     # n_actions = env.action_space.n
#     noise = 0
#     actions = []
#     from collections import deque

#     running_epr = deque([0 for i in range(101)],  maxlen=100)
#     for i_episode in range(N_eps):
        
#         observation = env.reset()
#         total_r = 0
#         done = False

#         for t in range(300):
#             T += 1

#             if T < warmup_steps:
#                 action = env.action_space.sample()[0]
#             else:
#                 curr_epsilon = scheduler.value(T - warmup_steps)
#                 noise = np.random.normal(0, curr_epsilon)
#                 if i_episode % 50 == 0 and t == 0 :
#                     print(f"noise of episode: {i_episode}, {noise}, epsilon: {curr_epsilon}")
#                 action_mean = policy_net(torch.from_numpy(observation).float())
#                 action = np.clip(action_mean.item() + noise , -2.0, 2.0)
#                 # print(action)
#             next_observation, reward, done, info = env.step([action])
#             total_r += reward
#             reward = torch.tensor([reward])
            
#             memory.push(torch.from_numpy(observation).view(1, -1), \
#                 action, reward, torch.from_numpy(next_observation).view(1, -1), float(done))
            
#             # train the DQN
#             if T % train_freq == 0:
#                 train_on_batch(memory, min(batch_size, T), df, T)

#             observation = next_observation

#             if done:
#                 break

#         running_epr.append(total_r)
#         if (i_episode + 1) % 100 == 0:
#             # print('curr eps', noise)
#             print("Episode {} finished with {} total rewards, T: {}".format(i_episode, np.mean(running_epr), T))
                
#         rewards.append(total_r)
        
#     for i in range(5):
#         observation = env.reset()
#         for j in range(500):
#             action_mean = policy_net(torch.from_numpy(observation).float())
#             action = np.clip(action_mean.item(), -2.0, 2.0)
#             next_observation, reward, done, info = env.step([action])
#             env.render()
#     env.close()
    
#     return rewards

if __name__ == '__main__':
    N_EPS = 10000
    args = get_args()
    # rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
    rewards_DDPG = learn_episodic_DDPG(args)
    # plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
    # plt.legend()
    # plt.show()
        

