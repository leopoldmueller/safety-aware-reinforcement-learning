# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
from visualizer import visualize_safety_layer
# ---------------------------------------------CODE------------------------------------------------------------------- #
env_name = 'CartPole-v1'
granularity = 5
max_depth = None
time_steps = 20000
num_episodes_to_show = 25
time_steps_test = 15000
create_new_log_files = True

supp_a = 0.0
supp_f = 0.005

conf_a = 1.0
conf_f = 0.95
# -------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    visualize_safety_layer(env_name=env_name,
                           granularity=granularity,
                           max_depth=max_depth,
                           time_steps=time_steps,
                           num_episodes_to_show=num_episodes_to_show,
                           time_steps_test=time_steps_test,
                           create_new_log_files=create_new_log_files,
                           supp_a=supp_a,
                           supp_f=supp_f,
                           conf_a=conf_a,
                           conf_f=conf_f)
# -------------------------------------------------------------------------------------------------------------------- #
