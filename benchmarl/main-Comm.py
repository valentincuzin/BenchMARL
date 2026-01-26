from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig, Callback
from benchmarl.models import MlpConfig, GnnConfig, SequenceModelConfig

from torch_geometric.nn import GCNConv, GATv2Conv, Sequential

if __name__ == "__main__":
    # Loads from "benchmarl/conf/algorithm/masac.yaml"
    algorithm_config = IppoConfig.get_from_yaml()

    # Some basic other configs
    experiment_config = ExperimentConfig.get_from_yaml()
    task = VmasTask.NAVIGATION.get_from_yaml()
    gnn_module_config = GnnConfig(
        topology="from_pos",
        self_loops=False,
        gnn_class=GCNConv,
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True,
        edge_radius=1,
    )
    model_config = SequenceModelConfig([gnn_module_config, MlpConfig.get_from_yaml()], [256])
    critic_model_config = MlpConfig.get_from_yaml()

    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=0,
        config=experiment_config,
    )
    experiment.run()
