# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
from utils import create_env, get_discrete_action, metric_dict, get_spaces, check_for_continuous_action_space, get_feature_names_n_classes
from dtreeviz.trees import *
# ---------------------------------------------CODE------------------------------------------------------------------- #


def find_constraints(max_depth=None, extract_rules=True, supp=0.0000, show_tree=True, granularity=5,
                     env_name="CartPole-v1", conf=1.0, lift=0.0, time_steps=10000, save_constraints=True):

    env = create_env(env_name=env_name)
    continuous_action = check_for_continuous_action_space(env=env)
    action_space, observation_space = get_spaces(env=env)

    path_root_trajectories = 'output\\trajectories\\{}\\time_steps_{}\\'.format(env_name, time_steps)
    path_root_rules = 'output\\constraints\\{}\\granularity_{}\\' \
                      'max_depth_{}\\supp_{}_conf_{}_lift_{}'.format(env_name, granularity, max_depth, supp, conf, lift)

    if save_constraints:
        if not os.path.exists(path_root_rules):
            os.makedirs(path_root_rules)

    # for each continuous action
    for m in range(action_space):

        # input paths
        path_obs = path_root_trajectories + '\\obs.csv'
        path_actions = path_root_trajectories + '\\actions.csv'

        # output paths
        path_statistic = path_root_rules + '\\statistic_action_{}.csv'.format(m)
        path_tree = path_root_rules + '\\tree_action_{}'.format(m)
        path_rule_sets = 'output\\constraints\\{}\\granularity_{}\\max_depth_{}\\rule_sets_{}.csv'.format(env_name, granularity, max_depth, m)

        # if there is no rules_m.csv, create one
        if not os.path.exists(path_rule_sets):
            df_rules = pd.DataFrame(columns=['supp', 'conf', 'lift', 'rule_num', 'rule_len', 'rule_mean_len'])
            df_rules.to_csv(path_or_buf=path_rule_sets, sep=';', index=False)

        df_rules = pd.read_csv(path_rule_sets, delimiter=';')

        # get number of classes and feature_names
        feature_names, n_classes = get_feature_names_n_classes(observation_space=observation_space,
                                                               action_space=action_space,
                                                               granularity=granularity,
                                                               continuous_action=continuous_action)

        # read input (observations and actions) from csv. into array
        df_actions = pd.read_csv(path_actions, delimiter=';', header=None)
        actions_array = df_actions[m].to_numpy()
        obs_array = np.genfromtxt(path_obs, delimiter=';')

        if continuous_action:
            for i in range(len(actions_array)):
                actions_array[i] = get_discrete_action(start=env.action_space.low[m],
                                                       stop=env.action_space.high[m],
                                                       num=granularity,
                                                       con_action=actions_array[i])

        # get unique actions
        actions_unique = np.unique(actions_array)

        # count frequency for each unique action
        y_absolute = []

        # relative frequency for each unique action
        y_relative = []

        # metrics ("all" is for each action in actions_array)
        # frequency of rule
        count_x = []
        # frequency of action (the action that fits the rule)
        count_y = []
        # frequency of each action in actions_array
        count_y_all = []
        # frequency of (rule => action)
        count_x_y = []
        # relative frequency of rule
        support_x = []
        # relative frequency of action (the action that fits the rule)
        support_y = []
        # relative frequency of each action in actions_array
        support_y_all = []
        # relative frequency of (rule => action)
        support_x_y = []
        # confidence(rule => action)
        confidence_x_y = []
        # lift(rule => action)
        lift_x_y = []

        observations_array = []
        features_array = []
        comparative_sign_array = []

        # get len(support_y_all)=len(count_y_all)=len(actions_array)
        for a in actions_array:
            support_y_all.append(a)
            count_y_all.append(a)

        # get frequency of each unique_action (append to y_absolute)
        for action in actions_unique:
            y_absolute.append(np.count_nonzero(actions_array == action))

        # get relative frequency of each unique_action (append to y_absolute)
        for number in y_absolute:
            y_relative.append(number / len(actions_array))

        for j in range(len(actions_unique)):
            for i in range(len(actions_array)):
                if support_y_all[i] == actions_unique[j]:
                    support_y_all[i] = y_relative[j]
                    count_y_all[i] = y_absolute[j]

        if supp==0.0:
            supp = 1

        # create classifier
        clf = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=supp, max_depth=max_depth)
        # fit input data
        clf = clf.fit(X=obs_array, y=actions_array)

        if show_tree:
            viz = dtreeviz(clf,
                           obs_array,
                           actions_array,
                           feature_names=feature_names)
            viz.view()

        if extract_rules:
            children_left = clf.tree_.children_left
            children_right = clf.tree_.children_right
            feature = clf.tree_.feature
            threshold = clf.tree_.threshold

            def find_path(node_numb, path, x):
                path.append(node_numb)
                if node_numb == x:
                    return True
                left = False
                right = False
                if children_left[node_numb] != -1:
                    left = find_path(children_left[node_numb], path, x)
                if children_right[node_numb] != -1:
                    right = find_path(children_right[node_numb], path, x)
                if left or right:
                    return True
                path.remove(node_numb)
                return False

            def get_rule(path, column_names):
                mask = ''
                observations = []
                features = []
                comparative_signs = []
                for index, node in enumerate(path):
                    # We check if we are not in the leaf
                    if index != len(path) - 1:
                        # Do we go under or over the threshold ?
                        if children_left[node] == path[index + 1]:
                            comparative_signs.append(1)
                            features.append(feature[node])
                            observations.append(threshold[node])
                            mask += "{} <= {} \t ".format(column_names[feature[node]], threshold[node])
                        else:
                            features.append(feature[node])
                            observations.append(threshold[node])
                            comparative_signs.append(0)
                            mask += "{} > {} \t ".format(column_names[feature[node]], threshold[node])
                # We insert the & at the right places
                mask = mask.replace('\t', 'and', mask.count('\t') - 1)
                mask = mask.replace('\t', '')
                return mask, observations, features, comparative_signs

            # Leaves
            leave_id = clf.apply(obs_array)

            paths = {}

            for leaf in np.unique(leave_id):
                path_leaf = []
                find_path(0, path_leaf, leaf)
                paths[leaf] = np.unique(np.sort(path_leaf))

            for p in paths:
                for k in range(len(paths[p])):
                    if (np.max(clf.tree_.value[paths[p][k]]) / clf.tree_.n_node_samples[paths[p][k]]) > conf:
                        paths[p] = paths[p][:k+1]
                        break

            # rename keys in path dictionary/ make paths unique
            for p in np.unique(leave_id):
                k = paths[p][-1]
                paths[k] = paths.pop(p)

            rules = {}

            for key in paths:
                rules[key], o, f, c = get_rule(paths[key], feature_names)
                observations_array.append(o)
                features_array.append(f)
                comparative_sign_array.append(c)

            actions = []

            for path in paths:
                actions.append(np.argmax(clf.tree_.value[paths[path][-1]]))
                count_x.append(clf.tree_.n_node_samples[paths[path][-1]])
                count_y.append(y_absolute[np.argmax(clf.tree_.value[paths[path][-1]])])
                support_y.append(y_relative[np.argmax(clf.tree_.value[paths[path][-1]])])
                count_x_y.append(np.max(clf.tree_.value[paths[path][-1]]))
                support_x_y.append(np.max(clf.tree_.value[paths[path][-1]]) / len(actions_array))

            for c in count_x:
                support_x.append(c / sum(count_x))

            for j in range(len(support_x_y)):
                confidence_x_y.append(support_x_y[j] / support_x[j])
                lift_x_y.append(confidence_x_y[j] / support_y[j])

            i = 0
            for rule in rules:
                rules[rule] = rules[rule][:-2]
                rules[rule] = 'elif ' + rules[rule] + ':' + ' action = ' + str(actions[i])
                i = i + 1

            # create data frame (constraints and metrics)
            df_statistic = pd.DataFrame.from_dict(data=rules, orient='index', columns=['constraints'])
            df_statistic['observations'] = observations_array
            df_statistic['features'] = features_array
            df_statistic['comparisons'] = comparative_sign_array
            df_statistic['actions'] = actions
            df_statistic['frequency(action)'] = count_y
            df_statistic['frequency(constraint)'] = count_x
            df_statistic['support(constraint)'] = support_x
            df_statistic['support(action)'] = support_y
            df_statistic['frequency(constraint=>action)'] = count_x_y
            df_statistic['support(constraint=>action)'] = support_x_y
            df_statistic['confidence(constraint=>action)'] = confidence_x_y
            df_statistic['lift(constraint=>action)'] = lift_x_y

            df_statistic = df_statistic.reset_index(drop=True)

            if supp == 1:
                supp = 0

            # make metrics iterable
            value_dict = {0: conf,
                          1: lift,
                          2: supp}

            # for each metric filter constraints
            for i in range(3):
                # get index to filter by metric
                rem_index = np.where(value_dict[i] > df_statistic[metric_dict[i]])

                # cut data type
                rem_index = rem_index[0]

                # filter by metric
                df_statistic = df_statistic.drop(rem_index)

                # reset index
                df_statistic = df_statistic.reset_index(drop=True)

            if len(df_statistic['constraints']) > 0:
                # first constraint is a if clause
                # df_statistic['constraints'][0] = df_statistic['constraints'].iloc[0][2:]
                df_statistic.loc[0, 'constraints'] = df_statistic.loc[0, 'constraints'][2:]

            if save_constraints:
                # save statistic as csv
                df_statistic.to_csv(path_or_buf=path_statistic, sep=';', index=False)

            rule_len = []
            for i in range(len(df_statistic['constraints'])):
                rule_len.append(len(df_statistic['features'][i]))

            # create data for evaluation of constraints identifier (visualize impact of metrics on rule set)
            if len(rule_len) == 0:
                rule_len_mean = 0
            else:
                rule_len_mean = np.mean(rule_len)

            # if there is no rule, set number of rules to 0
            if rule_len_mean == 0:
                number_of_rules = 0
            else:
                number_of_rules = len(df_statistic['constraints'])

            # add new rule set to df_rules
            df_rules.loc[len(df_rules.index)] = [supp, conf, lift, number_of_rules, rule_len, rule_len_mean]
            df_rules.to_csv(path_or_buf=path_rule_sets, sep=';', index=False)

            # print results
            print('Rule Set with:\n'
                  'Support:           {}\n'
                  'Confidence:        {}\n'
                  'Number of Rules:   {}\n'
                  'Mean Rule length:  {}\n'.format(supp, conf, number_of_rules, rule_len_mean))


if __name__ == "__main__":
    find_constraints(env_name='LunarLander-v2', max_depth=None, conf=1.0, supp=0.00, show_tree=False, time_steps=20000)
    find_constraints(env_name='LunarLander-v2', max_depth=None, conf=0.95, supp=0.005, show_tree=False, time_steps=20000)
