# ---------------------------------------------IMPORTS---------------------------------------------------------------- #
from visualizer import visualize_impact_of_metrics
# ---------------------------------------------CODE------------------------------------------------------------------- #

env_name = 'CartPole-v1'
granularity = 5
max_depth = None
action_index_m = 0
max_rules = 25

rules_already_exist = False
time_steps = 10000
# -------------------------------------------------------------------------------------------------------------------- #

if __name__ == "__main__":
    visualize_impact_of_metrics(granularity=granularity,
                                max_depth=max_depth,
                                env_name=env_name,
                                max_rules=max_rules,
                                action_index_m=action_index_m,
                                time_steps=time_steps,
                                rules_already_exist=rules_already_exist)
# -------------------------------------------------------------------------------------------------------------------- #
