import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from ast import literal_eval


def clean(x):
    return literal_eval(x)


metric_dict = {0: 'confidence(constraint=>action)',
               1: 'lift(constraint=>action)',
               2: 'support(constraint=>action)'}


def get_discrete_action(start, stop, num, con_action):
    discrete_actions = np.around(np.linspace(start=start, stop=stop, num=num), 2)
    disc_action = min(discrete_actions, key=lambda x: abs(x-con_action))
    return disc_action


def test_model(model, env, num_episodes=None, time_steps=None, render=None):
    if num_episodes is None:
        num_episodes = int(input('Enter num_episodes:'))
    if time_steps is None:
        time_steps = int(input('Enter time_steps:'))
    if render is None:
        if 'y' == input('Do you want to render? (y/n):'):
            render = True
        else:
            render = False

    for episode in range(num_episodes):
        total_reward = 0
        obs = env.reset()
        for t in range(time_steps):
            if render:
                env.render()
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print('Episode ' + str(episode + 1) + ' finished after {} time steps'.format(t + 1))
                print(total_reward)
                break
    env.close()


def create_env(env_name):
    env = make_vec_env(env_name, n_envs=1)
    return env


def get_spaces(env):
    if env.action_space.dtype == 'int64':
        action_space = 1
    elif env.action_space.dtype == 'float32':
        action_space = env.action_space.shape[0]
    else:
        print('Could not identify action_space!')
    observation_space = env.observation_space.shape[0]
    return action_space, observation_space


def check_for_continuous_action_space(env):
    if env.action_space.dtype == 'int64':
        return False
    elif env.action_space.dtype == 'float32':
        return True


def get_feature_names_n_classes(observation_space, action_space, continuous_action, granularity):
    feature_names = []
    for i in range(observation_space):
        feature = 'obs[' + str(i) + ']'
        feature_names.append(feature)
    if continuous_action:
        n_classes = granularity
    else:
        n_classes = action_space
    return feature_names, n_classes


def get_actions_dict(granularity, low, high):
    actions_dict = {}
    actions = np.linspace(low, high, num=granularity)
    for i in range(len(actions)):
        actions_dict[i] = actions[i]
    return actions_dict


def load_model(env_name):
    env = create_env(env_name)
    path_model = 'output\\models\\' + env_name + '_model'
    model = PPO.load(path_model, env, device='cuda')
    return model


def test_model(model, env, num_episodes=5, render=True):
    for episode in range(num_episodes):
        total_reward = 0
        t = 0
        done = False
        obs = env.reset()
        while not done:
            if render:
                env.render()
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            t += 1
            total_reward += reward
            if done:
                print('Episode ' + str(episode + 1) + ' finished after {} time steps'.format(t + 1))
                print(int(total_reward))
                break
    env.close()

