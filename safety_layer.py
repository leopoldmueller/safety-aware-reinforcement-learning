# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
import numpy as np
import pandas as pd
from utils import create_env, get_discrete_action, get_spaces, check_for_continuous_action_space, get_actions_dict, clean
import os
from stable_baselines3 import PPO
# ---------------------------------------------CODE------------------------------------------------------------------- #

########################################################################################################################


def get_counters(statistic_dict, action_space, session_num):
    for m in range(action_space):
        p = []
        n = []
        a = []
        ts = []
        for i in range(len(statistic_dict[m]['constraints'])):
            p.append(np.linspace(0, 0, num=session_num))
            n.append(np.linspace(0, 0, num=session_num))
            a.append(np.linspace(0, 0, num=session_num))
            ts.append(np.linspace(0, 0, num=session_num))
        statistic_dict[m]['count_p'] = p
        statistic_dict[m]['count_n'] = n
        statistic_dict[m]['count_a'] = a
        statistic_dict[m]['ts'] = ts

    return statistic_dict

########################################################################################################################


def check_continuous_action(statistic_dict, actions_dicts_array, con_action, obs, granularity,
                            session, env):
    dis_action = []
    for a in range(len(con_action[0])):
        dis_action.append(get_discrete_action(start=env.action_space.low[a],
                                              stop=env.action_space.high[a],
                                              num=granularity,
                                              con_action=con_action[0][a]))

    action = con_action

    for m in statistic_dict:
        for i in range(len(statistic_dict[m]['constraints'])):
            for o in range(len(statistic_dict[m]['observations'][i])):
                if statistic_dict[m]['comparisons'][i][o] == 1:
                    if obs[statistic_dict[m]['features'][i][o]] <= statistic_dict[m]['observations'][i][o]:
                        continue
                    else:
                        break
                if statistic_dict[m]['comparisons'][i][o] == 0:
                    if obs[statistic_dict[m]['features'][i][o]] > statistic_dict[m]['observations'][i][o]:
                        continue
                    else:
                        break
            else:
                statistic_dict[m]['count_a'][i][session] += 1
                if actions_dicts_array[m][statistic_dict[m]['actions'][i]] == dis_action[m]:
                    statistic_dict[m]['count_p'][i][session] += 1

                else:
                    statistic_dict[m]['count_n'][i][session] += 1

                action[0][m] = actions_dicts_array[m][statistic_dict[m]['actions'][i]]
                # stop as soon as a rule takes effect
                break

    return statistic_dict, action

########################################################################################################################


def check_discrete_action(dis_action, statistic_dict, obs, session):
    for m in statistic_dict:
        for i in range(len(statistic_dict[m]['constraints'])):
            for o in range(len(statistic_dict[m]['observations'][i])):
                if statistic_dict[m]['comparisons'][i][o] == 1:
                    if obs[statistic_dict[m]['features'][i][o]] <= statistic_dict[m]['observations'][i][o]:
                        continue
                    else:
                        break
                if statistic_dict[m]['comparisons'][i][o] == 0:
                    if obs[statistic_dict[m]['features'][i][o]] > statistic_dict[m]['observations'][i][o]:
                        continue
                    else:
                        break
            else:
                statistic_dict[m]['count_a'][i][session] += 1
                if statistic_dict[m]['actions'][i] == dis_action:
                    statistic_dict[m]['count_p'][i][session] += 1
                    action = [statistic_dict[m]['actions'][i]]
                else:
                    statistic_dict[m]['count_n'][i][session] += 1
                    action = [statistic_dict[m]['actions'][i]]
                # stop as soon as a rule takes effect
                break
        else:
            action = dis_action
    return statistic_dict, action

########################################################################################################################


def count_violations(env, model, session, path_check, time_steps_test, granularity, continuous_action,
                     make_log, render, ts_l, statistic_dict, unsafe):

    # path log file
    if unsafe:
        path_l = path_check + '\\session_{}_learn_{}_log_file_unsafe'.format(session, ts_l)
    else:
        path_l = path_check + '\\session_{}_learn_{}_log_file_safe'.format(session, ts_l)

    if continuous_action:
        # array with action_space of each action in action
        action_spaces = abs(env.action_space.high-env.action_space.low)
        action_space, observation_space = get_spaces(env=env)
        # create actions_array
        actions_dicts_array = []

        # get actions_dict
        for j in range(action_space):
            actions_dict = get_actions_dict(granularity=granularity, low=env.action_space.low[j], high=env.action_space.high[j])

            # for action m save possible actions (actions_dict) in actions_dicts_array
            actions_dicts_array.append(actions_dict)

        # arrays for output file (result)
        obs_array = []
        actions_array = []
        rewards_array = []
        episode_starts_array = []
        final_rewards_array = []

        # set time
        time = 0

        # set episode counter
        episode = 0

        # if time lower time_steps start new episode
        while time < time_steps_test:

            # reset done
            done = False

            # time steps of episode
            t = 0

            # reward of episode
            ep_reward = []

            # count episode
            episode += 1

            # get start state
            obs = env.reset()
            # add 1 if new episode starts
            episode_starts_array.append(1)

            # append first observation
            obs_array.append(obs)
            while not done:

                # render time step if render is true
                if render:
                    env.render()

                # get next action
                action, _states = model.predict(obs)

                if not unsafe:
                    # check next action
                    statistic_dict, action = check_continuous_action(statistic_dict=statistic_dict,
                                                                     actions_dicts_array=actions_dicts_array,
                                                                     con_action=action, obs=obs[0], env=env,
                                                                     granularity=granularity, session=session)

                # get next state, reward, done and info
                obs, reward, done, info = env.step(action)

                # step one time step
                t += 1

                # if time limit reached set done = True
                if time + t == time_steps_test:
                    done = True

                # if not done append next obs (obs=>actions)
                if not done:
                    obs_array.append(obs)

                    # append 0 if episode continues
                    episode_starts_array.append(0)

                    # append reward if episode continues
                    ep_reward.append(reward)

                    # append 0 if episode continues
                    final_rewards_array.append(0.0)

                # append action
                actions_array.append(action)

                # append reward
                rewards_array.append(reward)

                # if done print episode result
                if done:
                    ep_reward.append(reward)
                    final_rewards_array.append(sum(ep_reward)[0])
                    print('Episode ' + str(episode) +
                          ' finished after {} time steps with {} reward.'.format(t, sum(ep_reward)[0]))

                    # add time steps t to runtime time
                    time += t

        # close environment
        env.close()

        df_log = pd.DataFrame()
        df_log['obs'] = obs_array
        df_log['actions'] = actions_array
        df_log['episode_start'] = episode_starts_array
        df_log['rewards'] = rewards_array
        df_log['final_rewards'] = final_rewards_array

        if make_log:
            df_log.to_csv(path_or_buf=path_l + '.csv', index=False, sep=';')

    if not continuous_action:

        # arrays for output file (result)
        obs_array = []
        actions_array = []
        rewards_array = []
        episode_starts_array = []
        final_rewards_array = []

        # initialize action
        action = []

        # set time
        time = 0

        # set episode counter
        episode = 0

        # if time lower time_steps start new episode
        while time < time_steps_test:

            # reset done
            done = False

            # time steps of episode
            t = 0

            # reward of episode
            ep_reward = []

            # count episode
            episode += 1

            # get start state
            obs = env.reset()

            # add 1 if new episode starts
            episode_starts_array.append(1)

            # append first observation
            obs_array.append(obs)
            while not done:

                # render time step if render is true
                if render:
                    env.render()

                # get next action
                action, _states = model.predict(obs)

                if not unsafe:
                    # check action for violations
                    statistic_dict, action = check_discrete_action(dis_action=action,
                                                                   statistic_dict=statistic_dict,
                                                                   obs=obs[0],
                                                                   session=session)

                # get next state, reward, done and info
                obs, reward, done, info = env.step(action)

                # step one time step
                t += 1

                # if time limit reached set done = True
                if time + t == time_steps_test:
                    done = True

                # if not done append next obs (obs=>actions)
                if not done:
                    obs_array.append(obs)

                    # append 0 if episode continues
                    episode_starts_array.append(0)

                    # append reward if episode continues
                    ep_reward.append(reward)

                    # append 0 if episode continues
                    final_rewards_array.append(0.0)

                # append action
                actions_array.append(action)

                # append reward
                rewards_array.append(reward)

                # if done print episode result
                if done:
                    ep_reward.append(reward)
                    final_rewards_array.append(sum(ep_reward)[0])
                    print('Episode ' + str(episode) + ' finished after '
                                                      '{} time steps with {} reward.'.format(t, sum(ep_reward)[0]))

                    # add time steps t to runtime time
                    time += t

        # close environment
        env.close()

        df_log = pd.DataFrame()
        df_log['obs'] = obs_array
        df_log['actions'] = actions_array
        df_log['episode_start'] = episode_starts_array
        df_log['rewards'] = rewards_array
        df_log['final_rewards'] = final_rewards_array

        if make_log:
            df_log.to_csv(path_or_buf=path_l + '.csv', index=False, sep=';')

    return statistic_dict

########################################################################################################################


def check_rules(env_name="CartPole-v1", policy='MlpPolicy', granularity=4, max_depth=None, supp=0.0000, conf=1.0000,
                lift=0.0000, render=False, time_steps_learn=15000, time_steps_test=10000,
                time_steps_pre_train=0, session_num=5, make_log=True, unsafe=False):

    env = create_env(env_name=env_name)

    continuous_action = check_for_continuous_action_space(env=env)
    action_space, observation_space = get_spaces(env=env)

    # create new model (untrained)
    model = PPO(policy, env)

    path_rule_set_statistic = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}'.format(env_name,
                                                                                             granularity,
                                                                                             max_depth)

    path_rule_set_statistic_filename = '\\pre_learn_{}_learn_{}_test_{}_action_{}.csv'

    path_root = 'output\\constraints\\{}\\' \
                'granularity_{}\\max_depth_{}\\supp_{}_conf_{}_lift_{}'.format(env_name,
                                                                               granularity,
                                                                               max_depth,
                                                                               supp,
                                                                               conf,
                                                                               lift)

    path_check = path_root + '\\check_pre_learn_{}_learn_{}_test_{}'.format(time_steps_pre_train,
                                                                            time_steps_learn,
                                                                            time_steps_test)

    if not os.path.exists(path_check):
        os.makedirs(path_check)

    # initialize statistic dictionary and rule set dictionary
    statistic_dict = {}
    rule_set_dict = {}

    # make rule set dictionary
    for m in range(action_space):
        if not os.path.exists(path_rule_set_statistic + path_rule_set_statistic_filename.format(time_steps_pre_train,
                                        time_steps_learn,
                                        time_steps_test,
                                        m)):
            df_rules = pd.DataFrame(columns=['supp', 'conf', 'lift', 'rule_num', 'rule_len', 'rule_mean_len',
                                             'count_p', 'count_n', 'count_a', 'ts', 'count_p_rel', 'count_n_rel'])
            df_rules.to_csv(path_or_buf=path_rule_set_statistic + path_rule_set_statistic_filename.format(time_steps_pre_train,
                                        time_steps_learn,
                                        time_steps_test,
                                        m),
                            sep=';',
                            index=False)

        rule_set_dict[m] = pd.read_csv(path_rule_set_statistic + path_rule_set_statistic_filename.format(time_steps_pre_train,
                                        time_steps_learn,
                                        time_steps_test,
                                        m),
                                       delimiter=';',
                                       converters={'ts': clean, 'count_p_rel': clean, 'count_n_rel': clean,
                                                   'count_n': clean, 'count_p': clean, 'count_a': clean,
                                                   'rule_len': clean})

    # make statistic dictionary
    for m in range(action_space):
        statistic_dict[m] = pd.read_csv(path_root + '\\statistic_action_{}.csv'.format(m),
                                        delimiter=';',
                                        converters={'observations': clean, 'comparisons': clean, 'features': clean})

    # set ts_l = 0 (learned time steps)
    ts_l = 0
    ts_l_array = []

    # pre_train model
    model.learn(total_timesteps=time_steps_pre_train)
    ts_l += time_steps_pre_train
    ts_l_array.append(ts_l)

    # reset counters
    statistic_dict = get_counters(statistic_dict=statistic_dict, session_num=session_num+1, action_space=action_space)
    # test model and count violations
    statistic_dict = count_violations(env=env, model=model, session=0, path_check=path_check,
                                      time_steps_test=time_steps_test, granularity=granularity,
                                      continuous_action=continuous_action, make_log=make_log, render=render,
                                      ts_l=ts_l, statistic_dict=statistic_dict, unsafe=unsafe)

    for k in range(session_num):

        print('Session {} of {} started.\nStart to learn model...'.format(k+1, session_num))

        # learn model
        model.learn(total_timesteps=time_steps_learn)
        # add lear time steps to ts_l
        ts_l += time_steps_learn
        ts_l_array.append(ts_l)

        print('...learning done ({} time steps).\nStart testing model...'.format(ts_l))

        # test model and count violations
        statistic_dict = count_violations(env=env, model=model, session=k+1, path_check=path_check,
                                          time_steps_test=time_steps_test, granularity=granularity,
                                          continuous_action=continuous_action, make_log=make_log, render=render,
                                          ts_l=ts_l, statistic_dict=statistic_dict, unsafe=unsafe)

        print('...testing done.\nSession {} finished.'.format(k+1))

    else:

        # save model
        print('Save model...')
        model.save(path_check + '\\model_{}'.format(ts_l))
        print('...model saved.')

        for m in statistic_dict:
            p_rel = []
            n_rel = []
            ts = []
            p_c = []
            n_c = []
            a_c = []
            for k in range(len(statistic_dict[m]['constraints'])):
                p_r = []
                n_r = []
                p = []
                n = []
                a = []
                ts.append(ts_l_array)
                for j in range(len(statistic_dict[m]['count_p'][k])):
                    if statistic_dict[m]['count_a'][k][j] > 0:
                        p_r.append(statistic_dict[m]['count_p'][k][j]/statistic_dict[m]['count_a'][k][j])
                        n_r.append(statistic_dict[m]['count_n'][k][j]/statistic_dict[m]['count_a'][k][j])
                    else:
                        p_r.append(0)
                        n_r.append(0)
                    p.append(statistic_dict[m]['count_p'][k][j])
                    n.append(statistic_dict[m]['count_n'][k][j])
                    a.append(statistic_dict[m]['count_a'][k][j])

                p_rel.append(p_r)
                n_rel.append(n_r)
                p_c.append(p)
                n_c.append(n)
                a_c.append(a)

            statistic_dict[m]['count_p'] = p_c
            statistic_dict[m]['count_n'] = n_c
            statistic_dict[m]['count_a'] = a_c

            statistic_dict[m]['count_p_relative'] = p_rel
            statistic_dict[m]['count_n_relative'] = n_rel
            statistic_dict[m]['ts'] = ts

        # create rule_len for rule set statistic
        rule_len = []
        for m in statistic_dict:
            r_l = []
            for i in range(len(statistic_dict[m]['constraints'])):
                r_l.append(len(statistic_dict[m]['features'][i]))
            rule_len.append(r_l)

        count_p_rel_2 = []
        count_n_rel_2 = []
        count_p_2 = []
        count_n_2 = []
        count_a_2 = []

        for m in statistic_dict:
            count_p_rel = []
            count_n_rel = []
            count_p = []
            count_n = []
            count_a = []

            for j in range(len(ts_l_array)):
                sum_p = 0
                sum_n = 0
                sum_a = 0
                for i in range(len(statistic_dict[m]['constraints'])):
                    # sum_p_rel += statistic_dict[m]['count_p_relative'][i][j]
                    # sum_n_rel += statistic_dict[m]['count_n_relative'][i][j]
                    sum_p += statistic_dict[m]['count_p'][i][j]
                    sum_n += statistic_dict[m]['count_n'][i][j]
                    sum_a += statistic_dict[m]['count_a'][i][j]

                if sum_a != 0:
                    count_p_rel.append(sum_p / sum_a)
                    count_n_rel.append(sum_n / sum_a)
                else:
                    count_p_rel.append(0)
                    count_n_rel.append(0)
                count_p.append(sum_p)
                count_n.append(sum_n)
                count_a.append(sum_a)

            count_p_rel_2.append(count_p_rel)
            count_n_rel_2.append(count_n_rel)
            count_p_2.append(count_p)
            count_n_2.append(count_n)
            count_a_2.append(count_a)

        # append results to rule set statistic
        for m in rule_set_dict:
            rule_set_dict[m].loc[len(rule_set_dict[m].index)] = [supp, conf, lift,
                                                                 len(statistic_dict[m]['constraints']),
                                                                 rule_len[m], np.mean(rule_len[m]), count_p_2,
                                                                 count_n_2, count_a_2, ts_l_array, count_p_rel_2,
                                                                 count_n_rel_2]

        # save statistic and rule set statistic
        for m in statistic_dict:
            path_s = path_check + '\\learn_{}_statistic_action_{}'.format(ts_l, m)
            statistic_dict[m].to_csv(path_or_buf=path_s + '.csv', index=False, sep=';')
            rule_set_dict[m].to_csv(path_or_buf=path_rule_set_statistic + path_rule_set_statistic_filename.format(time_steps_pre_train,
                                        time_steps_learn,
                                        time_steps_test,
                                        m),
                                    index=False, sep=';')

if __name__ == "__main__":
    check_rules(env_name="CartPole-v1", granularity=5, supp=0.0050, conf=0.9, lift=0.0000, render=False,
                time_steps_test=8000, time_steps_learn=0, session_num=0, make_log=True, time_steps_pre_train=0,
                unsafe=False)