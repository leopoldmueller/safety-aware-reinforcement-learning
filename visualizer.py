# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
from safety_layer import check_rules
from utils import clean
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.interpolate as interp
from constraints_identifier import find_constraints
# ---------------------------------------------CODE------------------------------------------------------------------- #


def visualize_safety_layer(env_name, granularity, max_depth, time_steps, num_episodes_to_show, time_steps_test,
                           create_new_log_files, supp_a, supp_f, conf_a, conf_f, lift_a=0.0, lift_f=0.0):
    if create_new_log_files:
        check_rules(env_name=env_name, granularity=granularity, time_steps_learn=0, time_steps_test=time_steps_test,
                    session_num=0, supp=supp_a, conf=conf_a, unsafe=True)
        check_rules(env_name=env_name, granularity=granularity, time_steps_learn=0, time_steps_test=time_steps_test,
                    session_num=0, supp=supp_a, conf=conf_a, unsafe=False)
        check_rules(env_name=env_name, granularity=granularity, time_steps_learn=0, time_steps_test=time_steps_test,
                    session_num=0, supp=supp_f, conf=conf_f, unsafe=False)

    root_supervised = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}\\supp_{}_conf_{}_lift_{}\\' \
                 'check_pre_learn_0_learn_0_test_{}\\' \
                 'session_0_learn_0_log_file_safe.csv'
    root_unsupervised = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}\\supp_{}_conf_{}_lift_{}\\' \
                 'check_pre_learn_0_learn_0_test_{}\\' \
                 'session_0_learn_0_log_file_unsafe.csv'

    path_a_s_a = root_supervised.format(env_name, granularity, max_depth, supp_a, conf_a, lift_a, time_steps_test)

    path_a_s_f = root_supervised.format(env_name, granularity, max_depth, supp_f, conf_f, lift_f, time_steps_test)

    path_a_u = root_unsupervised.format(env_name, granularity, max_depth, supp_a, conf_a, lift_a, time_steps_test)

    path_e = 'output\\trajectories\\{}\\time_steps_{}\\log_file.csv'.format(env_name, time_steps)

    df_a_s_a = pd.read_csv(filepath_or_buffer=path_a_s_a, sep=';')

    df_a_s_f = pd.read_csv(filepath_or_buffer=path_a_s_f, sep=';')

    df_a_u = pd.read_csv(filepath_or_buffer=path_a_u, sep=';')

    df_e = pd.read_csv(filepath_or_buffer=path_e, sep=';', converters={'final_rewards': clean})

    for i in range(len(df_e['final_rewards'])):
        # df_e['final_rewards'][i] = df_e['final_rewards'][i][0]
        df_e.loc[i, 'final_rewards'] = df_e.loc[i, 'final_rewards'][0]

    df_e = df_e[df_e['final_rewards'] != 0]
    df_a_s_f = df_a_s_f[df_a_s_f['final_rewards'] != 0]
    df_a_u = df_a_u[df_a_u['final_rewards'] != 0]
    df_a_s_a = df_a_s_a[df_a_s_a['final_rewards'] != 0]

    df_e = df_e.reset_index(drop=True)
    df_a_s_f = df_a_s_f.reset_index(drop=True)
    df_a_u = df_a_u.reset_index(drop=True)
    df_a_s_a = df_a_s_a.reset_index(drop=True)

    df_e = df_e[0:num_episodes_to_show]
    df_a_s_f = df_a_s_f[0:num_episodes_to_show]
    df_a_u = df_a_u[0:num_episodes_to_show]
    df_a_s_a = df_a_s_a[0:num_episodes_to_show]

    x = np.arange(1, num_episodes_to_show+1, 1)

    df_e_mean = df_e['final_rewards'].mean()
    df_a_s_a_mean = df_a_s_a['final_rewards'].mean()
    df_a_s_f_mean = df_a_s_f['final_rewards'].mean()
    df_a_u_mean = df_a_u['final_rewards'].mean()

    print('Summe:')
    print(sum(df_e['final_rewards']), sum(df_a_s_a['final_rewards']), sum(df_a_s_f['final_rewards']), sum(df_a_u['final_rewards']))
    print('Min:')
    print(min(df_e['final_rewards']), min(df_a_s_a['final_rewards']), min(df_a_s_f['final_rewards']), min(df_a_u['final_rewards']))
    print('Max:')
    print(max(df_e['final_rewards']), max(df_a_s_a['final_rewards']), max(df_a_s_f['final_rewards']), max(df_a_u['final_rewards']))
    print('Anzahl:')
    print(len(df_e['final_rewards']), len(df_a_s_a['final_rewards']), len(df_a_s_f['final_rewards']), len(df_a_u['final_rewards']))

    print('Mittelwert:')
    print(df_e_mean, df_a_s_a_mean, df_a_s_f_mean, df_a_u_mean)

    df_e_var = df_e['final_rewards'].var()
    df_a_s_a_var = df_a_s_a['final_rewards'].var()
    df_a_s_f_var = df_a_s_f['final_rewards'].var()
    df_a_u_var = df_a_u['final_rewards'].var()

    print('Varianz')
    print(df_e_var, df_a_s_a_var, df_a_s_f_var, df_a_u_var)

    df_e_std = df_e['final_rewards'].std()
    df_a_s_a_std = df_a_s_a['final_rewards'].std()
    df_a_s_f_std = df_a_s_f['final_rewards'].std()
    df_a_u_std = df_a_u['final_rewards'].std()

    print('Standardabweichung')
    print(df_e_std, df_a_s_a_std, df_a_s_f_std, df_a_u_std)

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=x, y=df_e['final_rewards'], name='Expert',
                             line=dict(color='#2ca02c', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df_a_s_a['final_rewards'], name = 'Novice (unfiltered rule set)',
                             line=dict(color='#d62728', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df_a_s_f['final_rewards'], name='Novice (filtered rule set)',
                             line=dict(color='#ff7f0e', width=4)))
    fig.add_trace(go.Scatter(x=x, y=df_a_u['final_rewards'], name='Novice unsupervised',
                             line=dict(color='#17becf', width=4)))

    # Edit the layout
    fig.update_layout(title='{}'.format(env_name),
                      xaxis_title='Episode',
                      yaxis_title='Reward at the end of an episode', font=dict(size=30))

    fig.show()
# -------------------------------------------------------------------------------------------------------------------- #


def visualize_impact_of_metrics(granularity, max_depth, env_name, max_rules, action_index_m, rules_already_exist,
                                time_steps):
    if not rules_already_exist:
        supp = np.around(np.arange(0, 0.55, 0.05), 2)
        conf = np.around(np.arange(0, 1.05, 0.05), 2)

        denominator = len(supp) * len(conf)
        progress_counter = 0

        for i in range(len(supp)):
            for j in range(len(conf)):
                find_constraints(max_depth=max_depth,
                                 extract_rules=True,
                                 show_tree=False,
                                 granularity=granularity,
                                 env_name=env_name,
                                 time_steps=time_steps,
                                 supp=supp[i],
                                 conf=conf[j],
                                 save_constraints=False)
                progress_counter += 1
                print('Progress:      {}%\n'.format(np.around((progress_counter / denominator) * 100, 4)))

    path_input = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}\\' \
                 'rule_sets_{}.csv'.format(env_name, granularity, max_depth, action_index_m)

    data = pd.read_csv(filepath_or_buffer=path_input, sep=';')

    x = data['supp']
    y = data['conf']
    z = data['rule_num']

    plotx, ploty, = np.meshgrid(x, y)
    plotz = interp.griddata((x, y), z, (plotx, ploty), method='linear')

    fig = go.Figure(go.Surface(z=plotz, x=plotx, y=ploty, showscale=True, ))

    fig.update_layout(width=1000, height=1000,
                      scene=dict(xaxis_title='Support',
                                 yaxis_title='Confidence',
                                 zaxis_title='Number of rules'))
    fig.show()

    data.loc[data.rule_num > 20, 'rule_num'] = max_rules

    x = data['supp']
    y = data['conf']
    z = data['rule_num']

    plotx, ploty, = np.meshgrid(x, y)
    plotz = interp.griddata((x, y), z, (plotx, ploty), method='linear')

    fig = go.Figure(go.Surface(z=plotz, x=plotx, y=ploty, showscale=True, ))

    fig.update_layout(width=1000, height=1000,
                      scene=dict(xaxis_title='Support',
                                 yaxis_title='Confidence',
                                 zaxis_title='Number of rules'))
    fig.show()

    data = pd.read_csv(filepath_or_buffer=path_input, sep=';')

    x = data['supp']
    y = data['conf']
    z = data['rule_mean_len']

    plotx, ploty, = np.meshgrid(x, y)
    plotz = interp.griddata((x, y), z, (plotx, ploty), method='linear')

    fig = go.Figure(go.Surface(z=plotz, x=plotx, y=ploty, showscale=True, ))

    fig.update_layout(width=1000, height=1000,
                      scene=dict(xaxis_title='Support',
                                 yaxis_title='Confidence',
                                 zaxis_title='Mean length of rules'))
    fig.show()
# -------------------------------------------------------------------------------------------------------------------- #


def visualize_violations_of_rules(env_name, granularity, max_depth, action_index_m, rule_violations_already_counted,
                                  supp_filtered, conf_filtered, time_steps_learn, time_steps_test,
                                  session_num, time_steps_pre_train):
    if not rule_violations_already_counted:
        check_rules(env_name=env_name,
                    granularity=granularity,
                    supp=0.00,
                    conf=1.00,
                    lift=0.0000,
                    render=False,
                    time_steps_test=time_steps_test,
                    time_steps_learn=time_steps_learn,
                    session_num=session_num,
                    make_log=False,
                    time_steps_pre_train=0,
                    unsafe=False)
        check_rules(env_name=env_name,
                    granularity=granularity,
                    supp=supp_filtered,
                    conf=conf_filtered,
                    lift=0.0000,
                    render=False,
                    time_steps_test=time_steps_test,
                    time_steps_learn=time_steps_learn,
                    session_num=session_num,
                    make_log=False,
                    time_steps_pre_train=0,
                    unsafe=False)

    path_input = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}\\' \
                 'pre_learn_{}_learn_{}_test_{}_action_{}.csv'.format(env_name,
                                                                                                   granularity,
                                                                                                   max_depth,
                                                                                                   time_steps_pre_train,
                                                                                                   time_steps_learn,
                                                                                                   time_steps_test,
                                                                                                   action_index_m)

    data = pd.read_csv(filepath_or_buffer=path_input, sep=';', converters={'ts': clean,
                                                                           'count_p_rel': clean,
                                                                           'count_n_rel': clean,
                                                                           'count_n': clean,
                                                                           'count_p': clean,
                                                                           'count_a': clean,
                                                                           'rule_len': clean,
                                                                           'conf': clean,
                                                                           'supp': clean})

    def get_indices(data, supp, conf):
        data_conf_filtered = data.loc[(data['conf'] == conf)]
        index = data_conf_filtered.index[data_conf_filtered['supp'] == supp][0]
        return index

    index_filtered = get_indices(data, supp_filtered, conf_filtered)
    index_unfiltered = get_indices(data, 0.0, 1.0)

    x = data['ts'][index_filtered]

    # filtered
    count_p = data['count_p'][index_filtered][0]
    count_a = data['count_a'][index_filtered][0]
    count_n = data['count_n'][index_filtered][0]

    # unfiltered
    count_p_2 = data['count_p'][index_unfiltered][0]
    count_a_2 = data['count_a'][index_unfiltered][0]
    count_n_2 = data['count_n'][index_unfiltered][0]

    count_p = np.array(count_p)
    count_a = np.array(count_a)
    count_n = np.array(count_n)

    count_p_2 = np.array(count_p_2)
    count_a_2 = np.array(count_a_2)
    count_n_2 = np.array(count_n_2)

    count_p_rel_2 = count_p_2/count_a_2
    count_n_rel_2 = count_n_2/count_a_2

    count_p_rel = count_p/count_a
    count_n_rel = count_n/count_a

    fig = go.Figure()
    # Create and style traces
    fig.add_trace(go.Scatter(x=x, y=count_p, name='Rule is observed  (filtered)',
                             line=dict(color='#2ca02c', width=4)))
    fig.add_trace(go.Scatter(x=x, y=count_n, name = 'Rule is not observed  (filtered)',
                             line=dict(color='#d62728', width=4)))
    fig.add_trace(go.Scatter(x=x, y=count_a, name='Rule applies to state (filtered)',
                             line=dict(color='#17becf', width=4)))

    fig.add_trace(go.Scatter(x=x, y=count_p_2, name='Rule is observed  (unfiltered)',
                             line=dict(color='#2ca02c', width=4, dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=count_n_2, name = 'Rule is not observed  (unfiltered)',
                             line=dict(color='#d62728', width=4, dash='dot')))
    fig.add_trace(go.Scatter(x=x, y=count_a_2, name='Rule applies to state (unfiltered)',
                             line=dict(color='#17becf', width=4, dash='dot')))

    # Edit the layout
    fig.update_layout(title='{}'.format(env_name),
                      xaxis_title='Training progress in time steps',
                      yaxis_title='Absolute frequency', font=dict(size=30))

    fig.show()


    fig2 = go.Figure()
    # Create and style traces
    fig2.add_trace(go.Scatter(x=x, y=count_p_rel, name='Rule is observed  (filtered)',
                              line=dict(color='#2ca02c', width=4)))
    fig2.add_trace(go.Scatter(x=x, y=count_n_rel, name = 'Rule is not observed  (filtered)',
                              line=dict(color='#d62728', width=4)))

    fig2.add_trace(go.Scatter(x=x, y=count_p_rel_2, name='Rule is observed  (unfiltered)',
                              line=dict(color='#2ca02c', width=4, dash='dot')))
    fig2.add_trace(go.Scatter(x=x, y=count_n_rel_2, name = 'Rule is not observed  (unfiltered)',
                              line=dict(color='#d62728', width=4, dash='dot')))

    # Edit the layout
    fig2.update_layout(title='{}'.format(env_name),
                       xaxis_title='Training progress in time steps', yaxis_title='Relative frequency', font=dict(size=30))

    fig2.show()
# -------------------------------------------------------------------------------------------------------------------- #
