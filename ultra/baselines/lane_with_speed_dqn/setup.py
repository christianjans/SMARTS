from setuptools import setup

setup(
    name="lane_with_speed_dqn",
    description="Ultra lane with continuous speed action DQN agent",
    version="0.1.0",
    packages=["lane_with_speed_dqn"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
