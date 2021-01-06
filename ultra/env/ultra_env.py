from itertools import cycle
import glob, yaml
from smarts.core.scenario import Scenario
from smarts.env.hiway_env import HiWayEnv
from ultra.baselines.adapter import BaselineAdapter
import numpy as np
from scipy.spatial import distance
import math
from sys import path

path.append("./ultra")
from ultra.utils.common import (
    get_closest_waypoint,
    get_path_to_goal,
    ego_social_safety,
)


class UltraEnv(HiWayEnv):
    def __init__(
        self,
        agent_specs,
        scenario_info,
        headless,
        timestep_sec,
        seed,
        eval_mode=False,
        ordered_scenarios=False,
    ):
        self.timestep_sec = timestep_sec
        self.headless = headless
        self.scenario_info = scenario_info
        self.scenarios = self.get_task(scenario_info[0], scenario_info[1])
        if not eval_mode:
            _scenarios = glob.glob(f"{self.scenarios['train']}")
        else:
            _scenarios = glob.glob(f"{self.scenarios['test']}")

        self.ultra_scores = BaselineAdapter()
        super().__init__(
            scenarios=_scenarios,
            agent_specs=agent_specs,
            headless=headless,
            timestep_sec=timestep_sec,
            seed=seed,
            visdom=False,
        )

        if ordered_scenarios:
            scenario_roots = []
            for root in _scenarios:
                if Scenario.is_valid_scenario(root):
                    # The case that this is a scenario root
                    scenario_roots.append(root)
                else:
                    # The case that there this is a directory of scenarios: find each of the roots
                    scenario_roots.extend(Scenario.discover_scenarios(root))
            # Also see `smarts.env.HiwayEnv`
            self._scenarios_iterator = cycle(
                Scenario.variations_for_all_scenario_roots(
                    scenario_roots, list(agent_specs.keys())
                )
            )

    def generate_logs(self, observation, highwayenv_score):
        ego_state = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )
        closest_wp, _ = get_closest_waypoint(
            num_lookahead=100,
            goal_path=path,
            ego_position=ego_state.position,
            ego_heading=ego_state.heading,
        )
        signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        linear_jerk = np.linalg.norm(ego_state.linear_jerk)
        angular_jerk = np.linalg.norm(ego_state.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_state.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.position)

        angle_error = closest_wp.relative_heading(
            ego_state.heading
        )  # relative heading radians [-pi, pi]

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        info = dict(
            position=ego_state.position,
            speed=ego_state.speed,
            steering=ego_state.steering,
            heading=ego_state.heading,
            dist_center=abs(ego_dist_center),
            start=start,
            goal=goal,
            closest_wp=closest_wp,
            events=observation.events,
            ego_num_violations=ego_num_violations,
            social_num_violations=social_num_violations,
            goal_dist=goal_dist,
            linear_jerk=np.linalg.norm(ego_state.linear_jerk),
            angular_jerk=np.linalg.norm(ego_state.angular_jerk),
            env_score=self.ultra_scores.reward_adapter(observation, highwayenv_score),
        )
        return info

    def get_task(self, task_id, task_level):
        with open("ultra/config.yaml", "r") as task_file:
            scenarios = yaml.safe_load(task_file)["tasks"]
            task = scenarios[f"task{task_id}"][task_level]
        return task

    @property
    def info(self):
        return {
            "scenario_info": self.scenario_info,
            "timestep_sec": self.timestep_sec,
            "headless": self.headless,
        }
