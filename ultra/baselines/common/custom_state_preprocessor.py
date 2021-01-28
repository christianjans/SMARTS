from ultra.baselines.common.state_preprocessor import StatePreprocessor


class CustomStatePreprocessor(StatePreprocessor):
    def __init__(self, speed_normalization):
        self._speed_normalization = speed_normalization

    def _preprocess_state(self, state):
        speed = StatePreprocessor._extract_ego_speed(state)
        normalized_speed = self._speed_normalization(speed)
        position = StatePreprocessor._extract_ego_position(state)
        steering = StatePreprocessor._extract_ego_steering(state)
        heading = StatePreprocessor._extract_ego_heading(state)

        return {
            "speed": normalized_speed,
            "position": position,
            "steering": steering,
            "heading": heading,
        }

    def _normalize_speed(self, speed):
        return speed / self._speed_normalization
