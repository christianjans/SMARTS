import torch
from torch import nn
import numpy as np
from ultra.baselines.lane_with_speed_dqn.lane_with_speed_dqn.network import *
from smarts.core.agent import Agent
from ultra.utils.common import merge_discrete_action_spaces, to_3d_action, to_2d_action
import pathlib, os
from ultra.baselines.dqn.dqn.policy import DQNPolicy
from ultra.baselines.lane_with_speed_dqn.lane_with_speed_dqn.network import DQNWithSocialEncoder
from ultra.baselines.lane_with_speed_dqn.lane_with_speed_dqn.explore import EpsilonExplore
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.common.state_preprocessor import *


class LaneWithSpeedDQNPolicy(DQNPolicy):
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

        # The discrete action space for the
        # 'LaneControllerWithContinuousSpeed' controller.
        discrete_action_spaces = [
            # Target speed.
            [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0],
            # Change lane (-1 = right, 0 = stay, 1 = left)
            # [-1, 0, 1]
            [-1.0, 0.0, 1.0]
        ]
        action_size = discrete_action_spaces
        self.merge_action_spaces = 0
        self.action_space_type = "lane"
        # self.to_real_action = lambda action: np.asarray([action[0], int(action[1])])
        # self.to_real_action = lambda action: np.asarray(action)
        # self.to_real_action = lambda action: [action[0], int(action[1])]
        self.to_real_action = lambda action: [action[0], int(action[1])]

        # Convert the elements of the discrete_action_spaces to lists.
        self.index2actions = [
            merge_discrete_action_spaces([each])[0] for each in action_size
        ]
        self.action2indexs = [
            merge_discrete_action_spaces([each])[1] for each in action_size
        ]
        self.num_actions = [len(e) for e in action_size]
        print("-------------------------")
        print(self.index2actions)
        print(self.action2indexs)
        print(self.num_actions)
        print("-------------------------")

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
        prev_action_size = int(policy_params["prev_action_size"])  # Correct action size.
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

        # Changed convert_action_func.
        self.state_preprocessor = StatePreprocessor(
            preprocess_state_func=preprocess_state,
            # convert_action_func=lambda action: np.asarray(action[0], int(action[1])),
            convert_action_func=lambda action: np.asarray(action),
            state_description=self.state_description
        )

        self.replay = ReplayBuffer(
            buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
            batch_size=int(policy_params["replay_buffer"]["batch_size"]),
            state_preprocessor=self.state_preprocessor,
            device_name=self.device_name,
        )

    def step(self, state, action, reward, next_state, done, others=None):
        # Don't treat timeout as done equal to True.
        max_steps_reached = state["events"].reached_max_episode_steps
        if max_steps_reached:
            done = False
        
        # We are given an action in the form [x, y]

        # Convert the action back into floats so that we can look it up in
        # the action2index table.
        action = np.asarray([action[0], float(action[1])])

        # Transform the action to [[x], [y]] if we are not supposed to merge
        # action spaces.
        _action = (
            [[e] for e in action] if not self.merge_action_spaces
            else [action.tolist()]
        )
        # Get the action index of each action. We are then left with our
        # action index in the form of [a, b].
        action_index = np.asarray(
            [
                action2index[str(e)]
                for action2index, e in zip(self.action2indexs, _action)
            ]
        )

        # Add this experience to the replay buffer.
        self.replay.add(
            state=state,
            action=action_index,
            reward=reward,
            next_state=next_state,
            done=done,
            others=others,
            social_capacity=self.social_capacity,
            observation_num_lookahead=self.observation_num_lookahead,
            social_vehicle_config=self.social_vehicle_config,
            prev_action=self.prev_action,
        )

        # Perform gradient descent on previous experiences if it's time.
        if (
            self.step_count % self.train_step == 0
            and len(self.replay) >= self.batch_size
            and (self.warmup is None or len(self.replay) >= self.warmup)
        ):
            out = self.learn()
            self.update_count += 1
        else:
            out = {}

        # Update target network if it's time.
        if self.target_update > 1 and self.step_count % self.target_update == 0:
            self.update_target_network()
        elif self.target_update < 1.0:
            self.soft_update(
                self.target_q_network, self.online_q_network, self.target_update
            )
        self.step_count += 1
        self.prev_action = action

        return out
