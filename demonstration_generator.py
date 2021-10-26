# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
import numpy as np
import pandas as pd
import os
from utils import load_model, create_env, test_model
from stable_baselines3 import PPO
# ---------------------------------------------CODE------------------------------------------------------------------- #


def gen_model(time_steps_train, env_name):
    log_path = 'output\\models\\LOG\\'
    path_model = 'output\\models\\' + env_name + '_model'
    env = create_env(env_name=env_name)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
    model.learn(total_timesteps=time_steps_train)
    model.save(path_model)
    return model


def train_model(env_name, time_steps_train):
    path_model = 'output\\models\\' + env_name + '_model'
    model = load_model(env_name=env_name)
    model.learn(total_timesteps=time_steps_train)
    model.save(path=path_model)


def evaluate_saved_model(env_name, num_episodes=5, render=True):
    model = load_model(env_name=env_name)
    env = create_env(env_name=env_name)

    test_model(model, env, num_episodes=num_episodes, render=render)


def gen_trajectories(env_name, time_steps=10000):

    model = load_model(env_name=env_name)
    env = create_env(env_name=env_name)

    path_root = 'output\\trajectories\\{}\\time_steps_{}\\'.format(env_name, time_steps)
    if not os.path.exists(path_root):
        os.makedirs(path_root)

    path_actions = path_root + 'actions'
    path_obs = path_root + 'obs'
    path_log = path_root + 'log_file'

    obs_array = []
    actions_array = []
    reward_array = []
    episode_starts_array = []
    final_reward_array = []

    # set time
    time = 0

    # set episode counter
    episode = 0

    # if time lower time_steps start new episode
    while time < time_steps:

        # reset episode arrays
        ep_obs = []
        ep_actions = []
        ep_rewards = []
        ep_episode_starts = []
        ep_final_rewards = []

        # reset done
        done = False

        # time steps of episode
        t = 0

        # count episode
        episode += 1

        # get start state
        obs = env.reset()

        ep_episode_starts.append(1)

        # append first observation
        ep_obs.append(obs[0])

        # # append 0 for final reward in time_step
        # ep_final_rewards.append(0)
        while not done:

            action, state = model.predict(obs)

            # get next state, reward, done and info
            obs, reward, done, infos = env.step(action)

            # step one time step
            t += 1

            # if time limit reached set done = True
            if time + t == time_steps:
                done = True

            # append action
            ep_actions.append(action[0])

            # append reward
            ep_rewards.append(reward)

            # if done print episode result
            if done:
                ep_final_rewards.append(sum(ep_rewards))
                print('Episode ' + str(episode) + ' finished after {}'
                                                  ' time steps with {} reward.'.format(t, sum(ep_rewards)))

                # add time steps t to runtime time
                time += t

                # add data
                obs_array = obs_array + ep_obs
                actions_array = actions_array + ep_actions
                episode_starts_array = episode_starts_array + ep_episode_starts
                reward_array = reward_array + ep_rewards
                final_reward_array = final_reward_array + ep_final_rewards

                # print if episode is added
                print('Episode {} added.'.format(episode))

            # if not done append next obs (obs=>actions)
            else:
                ep_obs.append(obs[0])
                ep_final_rewards.append([0.0])
                ep_episode_starts.append(0)

    # close environment
    env.close()

    df_log = pd.DataFrame()
    df_log['obs'] = obs_array
    df_log['actions'] = actions_array
    df_log['episode_start'] = episode_starts_array
    df_log['rewards'] = reward_array
    df_log['final_rewards'] = final_reward_array

    df_log.to_csv(path_or_buf=path_log + '.csv', index=False, sep=';')

    # make trajectories
    np.savetxt(path_actions + '.csv', delimiter=';', X=actions_array)
    np.savetxt(path_obs + '.csv', delimiter=';', X=obs_array)


if __name__ == "__main__":
    gen_trajectories(env_name='CartPole-v1', time_steps=20000)
    gen_trajectories(env_name='LunarLander-v2', time_steps=20000)
