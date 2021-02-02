from ultra.baselines.common.state_preprocessor import StatePreprocessor


class CustomStatePreprocessor(StatePreprocessor):
    """An example of how one would implement their own state preprocessor."""

    def __init__(self, speed_normalization):
        self._speed_normalization = speed_normalization

    def _preprocess_state(self, state, social_vehicle_config):
        speed = StatePreprocessor.extract_ego_speed(state)
        normalized_speed = self._speed_normalization(speed)
        position = StatePreprocessor.extract_ego_position(state)
        steering = StatePreprocessor.extract_ego_steering(state)
        heading = StatePreprocessor.extract_ego_heading(state)
        waypoints = StatePreprocessor.extract_ego_waypoints(state)
        social_vehicles = StatePreprocessor.extract_social_vehicles(state)

        social_vehicles = StatePreprocessor.get_social_vehicles(
            social_vehicles=social_vehicles,
            social_vehicle_config=social_vehicle_config,
            ego_position=position,
            ego_heading=heading,
            ego_waypoints=waypoints,
        )

        return {
            "ego_info": [normalized_speed, position, steering, heading],
            "social_vehicles_info": social_vehicles,
        }

    def _normalize_speed(self, speed):
        return speed / self._speed_normalization
