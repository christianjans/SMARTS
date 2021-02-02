import collections.abc
import numpy as np
import torch

from ultra.baselines.common.social_vehicle_extraction import get_social_vehicles
from ultra.baselines.common.state_preprocessor import StatePreprocessor
from ultra.utils.common import (
    rotate2d_vector,
    normalize_im,
)


class BaselineStatePreprocessor(StatePreprocessor):
    """The State Preprocessor used by the baseline agents."""

    _NORMALIZATION_REFERENCES = {
        "speed": 30.0,
        "distance_from_center": 1.0,
        "steering": 3.14,  # radians
        "angle_error": 3.14,  # radians
        "relative_goal_position": 100.0,
        "action": 1.0,  # 2
        "waypoints_lookahead": 10.0,
        "road_speed": 30.0,
    }

    def __init__(
        self, social_vehicle_config, observation_waypoints_lookahead, action_size,
    ):
        self._state_description = {
            "images": {},
            "low_dim_states": {
                "speed": 1,
                "distance_from_center": 1,
                "steering": 1,
                "angle_error": 1,
                "relative_goal_position": 2,
                "action": int(action_size),  # 2
                "waypoints_lookahead": 2 * int(observation_waypoints_lookahead),
                "road_speed": 1,
            },
            "social_vehicles": int(social_vehicle_config["num_social_features"])
            if int(social_vehicle_config["social_capacity"]) > 0
            else 0,
        }

    @property
    def num_low_dim_states(self):
        return sum(self._state_description["low_dim_states"].values())

    def _preprocess_state(
        self,
        state,
        observation_num_lookahead,
        social_capacity,
        social_vehicle_config,
        prev_action,
        normalize=False,
        unsqueeze=False,
        device=None,
    ):
        # Obtain the information from the state needed by the baselines.
        state = self._adapt_observation_for_baseline(state)

        # Get the images from the state.
        images = {}
        for k in self._state_description["images"]:
            image = torch.from_numpy(state[k])
            image = image.unsqueeze(0) if unsqueeze else image
            image = image.to(device) if device else image
            image = normalize_im(image) if normalize else image
            images[k] = image

        # Set the action in the basic state as the previous action.
        state["action"] = prev_action

        # Obtain the next waypoints.
        _, lookahead_waypoints = self.extract_closest_waypoint(
            ego_goal_path=state["goal_path"],
            ego_position=state["ego_position"],
            ego_heading=state["heading"],
            num_lookahead=observation_num_lookahead,
        )
        state["waypoints_lookahead"] = np.hstack(lookahead_waypoints)

        # Normalize the low-dimensional states and concatenate them.
        normalized = [
            self._normalize(key, state[key])
            for key in self._state_description["low_dim_states"]
        ]
        low_dim_states = [
            value
            if isinstance(value, collections.abc.Iterable)
            else np.asarray([value]).astype(np.float32)
            for value in normalized
        ]
        low_dim_states = torch.cat(
            [torch.from_numpy(e).float() for e in low_dim_states], dim=-1
        )
        low_dim_states = low_dim_states.unsqueeze(0) if unsqueeze else low_dim_states
        low_dim_states = low_dim_states.to(device) if device else low_dim_states

        # Apply the social vehicle encoder (if applicable).
        state["social_vehicles"] = (
            get_social_vehicles(
                ego_vehicle_pos=state["ego_position"],
                ego_vehicle_heading=state["heading"],
                neighborhood_vehicles=state["social_vehicles"],
                social_vehicle_config=social_vehicle_config,
                waypoint_paths=state["waypoint_paths"],
            )
            if social_capacity > 0
            else []
        )
        social_vehicle_dimension = self._state_description["social_vehicles"]
        social_vehicles = torch.empty(0, 0)

        if social_vehicle_dimension:
            social_vehicles = torch.from_numpy(
                np.asarray(state["social_vehicles"])
            ).float()
            social_vehicles = social_vehicles.reshape((-1, social_vehicle_dimension))
        social_vehicles = social_vehicles.unsqueeze(0) if unsqueeze else social_vehicles
        social_vehicles = social_vehicles.to(device) if device else social_vehicles

        out = {
            "images": images,
            "low_dim_states": low_dim_states,
            "social_vehicles": social_vehicles,
        }
        return out

    @staticmethod
    def _adapt_observation_for_baseline(state):
        # Get basic information about the ego vehicle.
        ego_position = StatePreprocessor.extract_ego_position(state)
        ego_heading = StatePreprocessor.extract_ego_heading(state)
        ego_speed = StatePreprocessor.extract_ego_speed(state)
        ego_steering = StatePreprocessor.extract_ego_steering(state)
        ego_start = StatePreprocessor.extract_ego_start(state)
        ego_goal = StatePreprocessor.extract_ego_goal(state)
        ego_waypoints = StatePreprocessor.extract_ego_waypoints(state)
        social_vehicle_states = StatePreprocessor.extract_social_vehicles(state)

        # Identify the path the ego is following.
        ego_goal_path = StatePreprocessor.extract_ego_goal_path(
            ego_goal=ego_goal, ego_waypoints=ego_waypoints, ego_start=ego_start,
        )

        # Get the closest waypoint to the ego.
        ego_closest_waypoint, _ = StatePreprocessor.extract_closest_waypoint(
            ego_goal_path=ego_goal_path,
            ego_position=ego_position,
            ego_heading=ego_heading,
            num_lookahead=100,
        )

        # Calculate the ego's distance from the center of the lane.
        signed_distance_from_center = ego_closest_waypoint.signed_lateral_error(
            ego_position
        )
        lane_width = ego_closest_waypoint.lane_width * 0.5
        ego_distance_from_center = signed_distance_from_center / lane_width

        # Calculate the ego's relative, rotated position from the goal.
        ego_relative_rotated_goal_position = rotate2d_vector(
            np.asarray(ego_goal.position[0:2]) - np.asarray(ego_position[0:2]),
            -ego_heading,
        )

        basic_state = dict(
            speed=ego_speed,
            relative_goal_position=ego_relative_rotated_goal_position,
            distance_from_center=ego_distance_from_center,
            steering=ego_steering,
            angle_error=ego_closest_waypoint.relative_heading(ego_heading),
            social_vehicles=social_vehicle_states,
            road_speed=ego_closest_waypoint.speed_limit,
            start=ego_start.position,
            goal=ego_goal.position,
            heading=ego_heading,
            goal_path=ego_goal_path,
            ego_position=ego_position,
            waypoint_paths=ego_waypoints,
            events=state.events,
        )
        return basic_state

    @staticmethod
    def _normalize(key, value):
        if key not in BaselineStatePreprocessor._NORMALIZATION_REFERENCES:
            return value
        return value / BaselineStatePreprocessor._NORMALIZATION_REFERENCES[key]
