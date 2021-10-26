# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
from visualizer import visualize_violations_of_rules
# ---------------------------------------------CODE------------------------------------------------------------------- #
# set environment
env_name = 'CartPole-v1'
granularity = 5
max_depth = None
action_index_m = 0

# Parameters of the safety layer to be compared with unsupervised/ unfiltered Safety Layer
rule_violations_already_counted = False
supp_filtered = 0.005
conf_filtered = 0.9
time_steps_learn = 500
time_steps_test = 2000
session_num = 20
time_steps_pre_train = 0
# -------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    visualize_violations_of_rules(env_name=env_name, granularity=granularity, max_depth=max_depth,
                                  action_index_m=action_index_m,
                                  rule_violations_already_counted=rule_violations_already_counted,
                                  supp_filtered=supp_filtered,
                                  conf_filtered=conf_filtered,
                                  time_steps_learn=time_steps_learn,
                                  time_steps_test=time_steps_test,
                                  session_num=session_num,
                                  time_steps_pre_train=time_steps_pre_train)
# -------------------------------------------------------------------------------------------------------------------- #
