from setuptools import setup

setup(
    name="lane_dqn",
    description="Ultra lane action DQN agent",
    version="0.1.0",
    packages=["lane_dqn"],
    include_package_data=True,
    install_requires=["tensorflow==1.15", "smarts"],
)
