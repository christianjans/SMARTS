import torch
from torch import nn
import numpy as np
from ultra.baselines.lane_dqn.lane_dqn.network import *
from smarts.core.agent import Agent
from ultra.utils.common import merge_discrete_action_spaces, to_3d_action, to_2d_action
import pathlib, os
from ultra.baselines.dqn.dqn.policy import DQNPolicy
from ultra.baselines.lane_dqn.lane_dqn.network import DQNWithSocialEncoder
from ultra.baselines.lane_dqn.lane_dqn.explore import EpsilonExplore
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.common.state_preprocessor import *


class LaneDQNPolicy(DQNPolicy):
    def __init__(
        self, policy_params=None, checkpoint_dir=None,
    ):
        # Set the type of neural network we will use.
        network_class = DQNWithSocialEncoder

        self.epsilon_obj = EpsilonExplore(
            max_epsilon=1.0,
            min_epsilon=0.05,
            decay=100000
        )

        # The discrete action space for the 'Lane' controller.
        # [0] - "keep lane"
        # [1] - "slow down"
        # [2] - "change lane left"
        # [3] - "change lane right"
        discrete_action_spaces = [[0], [1], [2], [3]]
        action_size = discrete_action_spaces
        self.merge_action_spaces = -1
        self.action_space_type = "lane"
        self.to_real_action = lambda action: self.lane_actions[action[0]]

        # Convert the elements of the discrete_action_spaces to lists if
        # they are not already lists (i.e. a NumPy array).
        index_to_actions = [
            e.tolist() if not isinstance(e, list) else e for e in action_size
        ]
        # Converts each action in discrete_action_spaces to a string and
        # associates this key with its index in the list.
        # E.g. {'[0]': 0, '[1]': 1, '[2]': 2, '[3]': 3}
        # E.g. {'[0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0]': 0,
        #       '[-1, 0, 1]': 1}
        actions_to_indexs = {
            str(k): v
            for k, v in zip(
                index_to_actions, np.arange(len(index_to_actions)).astype(np.int)
            )
        }
        self.index2actions, self.action2indexs = (
            [index_to_actions],
            [actions_to_indexs]
        )
        self.num_actions = [len(index_to_actions)]

        self.step_count = 0
        self.update_count = 0
        self.num_updates = 0
        self.current_sticky = 0
        self.current_iteration = 0

        lr = float(policy_params["lr"])
        seed = int(policy_params["seed"])
        self.train_step = int(policy_params["train_step"])
        self.target_update = float(policy_params["target_update"])
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.warmup = int(policy_params["warmup"])
        self.gamma = float(policy_params["gamma"])
        self.batch_size = int(policy_params["batch_size"])
        self.use_ddqn = policy_params["use_ddqn"]
        self.sticky_actions = int(policy_params["sticky_actions"])
        prev_action_size = 1
        self.prev_action = np.zeros(prev_action_size)

        # State preprocessing.
        self.social_policy_hidden_units = int(
            policy_params["social_vehicles"].get("social_capacity_hidden_units", 0)
        )
        self.social_capacity = int(
            policy_params["social_vehicles"].get("social_capacity", 0)
        )
        self.observation_num_lookahead = int(
            policy_params.get("observation_num_lookahead", 0)
        )
        self.social_polciy_init_std = int(
            policy_params["social_vehicles"].get("social_polciy_init_std", 0)
        )
        self.num_social_features = int(
            policy_params["social_vehicles"].get("num_social_features", 0)
        )
        self.social_vehicle_config = get_social_vehicle_configs(
            **policy_params["social_vehicles"]
        )

        # State description.
        self.state_description = get_state_description(
            policy_params["social_vehicles"],
            policy_params["observation_num_lookahead"],
            prev_action_size
        )

        # Social vehicle encoder.
        self.social_vehicle_encoder = self.social_vehicle_config["encoder"]
        self.social_feature_encoder_class = self.social_vehicle_encoder[
            "social_feature_encoder_class"
        ]
        self.social_feature_encoder_params = self.social_vehicle_encoder[
            "social_feature_encoder_params"
        ]

        self.checkpoint_dir = checkpoint_dir

        self.reset()

        torch.manual_seed(seed)
        network_params = {
            "state_size": self.state_size,
            "social_feature_encoder_class": self.social_feature_encoder_class,
            "social_feature_encoder_params": self.social_feature_encoder_params,
        }
        self.online_q_network = network_class(
            num_actions=self.num_actions, **(network_params if network_params else {}),
        ).to(self.device)
        self.target_q_network = network_class(
            num_actions=self.num_actions, **(network_params if network_params else {}),
        ).to(self.device)
        self.update_target_network()

        self.optimizers = torch.optim.Adam(
            params=self.online_q_network.parameters(), lr=lr
        )
        self.loss_func = nn.MSELoss(reduction="none")

        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

        self.state_preprocessor = StatePreprocessor(
            preprocess_state_func=preprocess_state,
            convert_action_func=self.lane_action_to_index,
            state_description=self.state_description
        )

        self.replay = ReplayBuffer(
            buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
            batch_size=int(policy_params["replay_buffer"]["batch_size"]),
            state_preprocessor=self.state_preprocessor,
            device_name=self.device_name,
        )
