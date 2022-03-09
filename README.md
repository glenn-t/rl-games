# Reinforecment Learning solutions for Board Games

This project aims to produce reinforcement learning solutions for board games. It is currently in the early stages.

The game implementations are based on the [OpenSpiel framework](https://github.com/deepmind/open_spiel), a reinforcement learning framework for games.

Current games:

  * Tic Tac Toe (an OpenSpiel example).
  * [Dao](https://boardgamegeek.com/boardgame/948/dao) (in progress).

## Development Guide

For reproducibility reasons, I am using VS code development containers. This is a convenient way to develop code within a Docker enviornment.

Requirements:

  * Docker.
  * The OpenSpiel Docker container must be built on the local system, and tagged `openspiel`. See [here](https://github.com/deepmind/open_spiel/blob/master/docs/install.md) for details. Option 1 (Basic) contains more utilities so is recommended.
    + For reproducibility, commit `5354afc54d8cb1f96ebc320af16243ac6bdcb0cb` of the Open Spiel Github repository was used to build the docker image.
  * VS Code & [VS Code Remote Containers](https://code.visualstudio.com/docs/remote/containers) extension.

As an alternative, OpenSpiel now offers a pip package for users who only want to use the Python API. To do this:

  * Set up virtual environment: `python -m venv .venv`
  * Activate (on linux): `source .venv/bin/activate` 
  * Install packages: `pip3 install -r requirements_venv.txt`

The package is being updated to use open-spiel==1.1.0.
