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


## Things to explore

  * Seperate the idea of game state and player state - models can learn from player state instead.
  * Afterstate trainer - currently only learns on terminal rewards. Some games that have victory points could have intermediate rewards.

## Example Runs

```
uv run python -m agents.afterstate_trainer --selfplay 200000 --eval-every 5000 --output trained_models/w_table.npy

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 0.99 --seed 1 --output trained_models/w_099_s1.npy --mix 0

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 1.0 --seed 1 --output trained_models/w_100_s1.npy --mix 0

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 0.99 --seed 1 --output trained_models/w_099_s1_naive.npy --mix 1 --mix-agent naive

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive.npy --mix 1 --mix-agent naive

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 0.99 --seed 1 --output trained_models/w_099_s1_half_naive.npy --mix 0.5 --mix-agent naive

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_half_naive.npy --mix 0.5 --mix-agent naive

uv run python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 1.0 --epsilon-min 0.005 --gamma 0.99 --seed 1 --output trained_models/w_099_s1_half_naive_td.npy --mix 0.5 --mix-agent naive --update-method td
```