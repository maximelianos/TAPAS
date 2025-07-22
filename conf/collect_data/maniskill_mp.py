from conf._machine import data_naming_config
from conf.dataset.scene.maniskill import scene_dataset_config
from conf.env.maniskill.gmm import maniskill_env_config
from tapas_gmm.collect_data import Config
from tapas_gmm.env import Environment
from tapas_gmm.policy import PolicyEnum

config = Config(
    n_episodes=5,
    sequence_len=None,
    data_naming=data_naming_config,
    dataset_config=scene_dataset_config,
    env_type=Environment.MANISKILL,
    env_config=maniskill_env_config,
    policy_type=PolicyEnum.MOTION_PLANNER,
    policy=None,
)
