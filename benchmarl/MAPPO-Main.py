from benchmarl.algorithms import MappoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import ExperimentConfig
from benchmarl.benchmark import Benchmark
from benchmarl.models import MlpConfig

if __name__ == "__main__":
    algorithm_config = MappoConfig.get_from_yaml()
    algorithm_config.share_param_critic = True
    experiment_config = ExperimentConfig.get_from_yaml()
    experiment_config.share_policy_params = True

    nav = VmasTask.NAVIGATION.get_from_yaml()

    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    benchmark = Benchmark(
        algorithm_configs=[algorithm_config],
        model_config=model_config,
        critic_model_config=critic_model_config,
        experiment_config=experiment_config,
        tasks=[nav],
        seeds={0, 1, 2}
    )
    benchmark.run_sequential()
