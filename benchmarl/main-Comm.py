from benchmarl.algorithms import IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.benchmark import Benchmark
from benchmarl.models import MlpConfig, GnnConfig, SequenceModelConfig

from torch_geometric.nn import GATv2Conv

if __name__ == "__main__":
    algorithm_config = IppoConfig.get_from_yaml()
    algorithm_config.share_param_critic = False
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.share_policy_params = False

    navigation = VmasTask.NAVIGATION.get_from_yaml()
    joint_passage_size = VmasTask.JOINT_PASSAGE_SIZE.get_from_yaml()
    transport = VmasTask.TRANSPORT.get_from_yaml()

    gat_module_config = GnnConfig(
        topology="from_pos",
        self_loops=False,
        gnn_class=GATv2Conv,
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        exclude_pos_from_node_features=True,
        edge_radius=1,
    )
    gat_model_config = SequenceModelConfig([gat_module_config, MlpConfig.get_from_yaml()], [256])
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=[algorithm_config],
        model_config=gat_model_config,
        critic_model_config=critic_model_config,
        experiment_config=experiment_config,
        tasks=[navigation, joint_passage_size, transport],
        seeds={0, 1, 2}
    )
    benchmark.run_sequential()
