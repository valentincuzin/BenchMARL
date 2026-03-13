from benchmarl.algorithms import GppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.benchmark import Benchmark
from benchmarl.models import MlpConfig, GnnConfig

from torch_geometric.nn import GATv2Conv

if __name__ == "__main__":
    algorithm_config = GppoConfig.get_from_yaml()
    algorithm_config.share_param_critic = True
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.share_policy_params = True

    nav = VmasTask.NAVIGATION.get_from_yaml()
    comm_model_config = GnnConfig(
        bandwidth=32,
        topology="from_pos",
        self_loops=True,
        L_layers=2,
        gnn_class=GATv2Conv,
        position_key="pos",
        pos_features=2,
        velocity_key="vel",
        vel_features=2,
        edge_radius=1,
    )

    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=[algorithm_config],
        model_config=model_config,
        critic_model_config=critic_model_config,
        comm_model_config=comm_model_config,
        experiment_config=experiment_config,
        tasks=[nav],
        seeds={0, 1, 2}
    )
    benchmark.run_sequential()
