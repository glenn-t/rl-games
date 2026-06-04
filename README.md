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

  * Afterstate trainer - currently only learns on terminal rewards. Some games that have victory points could have intermediate rewards.
  * Use simple test game - tug of war - two players try and move a marker. Starts at 0, player 0 tries to get it to 10, player 1 tries to get it to -10. Each player takes turns moving the marker one step either way. Can test both terminal rewards and intermediate rewards (aka like a victory points game)

## Example Runs

```
python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1.npy --mix 0

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive.npy --mix 1 --mix-agent naive

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive_td.npy --mix 1 --mix-agent naive --update-method td

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.005 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_half_naive.npy --mix 0.5 --mix-agent naive

uv run python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_half_naive_td.npy --mix 0.5 --mix-agent naive --update-method td

python -m agents.afterstate_trainer --selfplay 2000000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_2m.npy --mix 0

python -m agents.afterstate_trainer --selfplay 2000000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_2m_td.npy --mix 0 --update-method td
```

Try with shaped rewards: -10 for loss. Observation is that TD learns faster than Monte Carlo, and Monte Carlo is terrible with heavily asymmetric rewards.

```
python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive_weighted.npy --mix 1.0 --mix-agent naive --update-method mc --eval-every 5000

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.2 --epsilon-min 0.001 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive_weighted_td.npy --mix 1.0 --mix-agent naive --update-method td --eval-every 5000
```

Try optimistic initial values (which is used if epsilon-start is 0)

```
python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.0 --epsilon-min 0.0 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive_weighted_eps_0.npy --mix 1.0 --mix-agent naive --update-method mc --eval-every 5000

python -m agents.afterstate_trainer --selfplay 200000 --alpha 0.1 --alpha-final 0.001 --epsilon-start 0.0 --epsilon-min 0.0 --gamma 1.0 --seed 1 --output trained_models/w_100_s1_naive_weighted_eps_0_td.npy --mix 1.0 --mix-agent naive --update-method td --eval-every 5000
```

Optimisitic values not working that well - but as below, even when playing as a single side.

python dao_test.py --player1 afterstate --player2 naive --num_games 200

## Analysis of issues

Example issues.

The issue is that this board state is ambigious - should you move to this state?

```
o o . o
x x . .
. x o .
x . . . 
```

You should actually never move to this state, as it means the opponent can win on the next turn.

Current symmetric learning approach
If X moved to this, W should shift to -1
If O moved to this, W should shift to 1

We have a broken assumption - if a state is bad for X, it's good for O. But this state is bad for both!

We can't have a symmetric learner. 

Better approach - have a learner that sees the board from its own perspective - e.g. always X - player rewards are given accordingly. The game is symmetric - it doesn't matter if I play as X or O, given all states can be revisited.
For play against a fixed agent - if playing as O, remap pieces to X and recalculate canonical states.
Same applies to self-play
Need the concept of a player_state and player_reward.

## Observations of the afterstate agent on Dao

  * TD is faster and better than MC
  * Pure self play (2m games) achieves near minimax performance (though haven't validated - 0 losses against naive agent). Also achieves higher average reward against an agent trained against naive agent only (but only for 200k as its slower) - that agent 

 Naive agent training only - 200k games, TD
```
 vs Random: 100.0%W/0.0%L/0.0%/1.0 | vs Naive: 99.6%W/0.4%L/0.0%/0.9582
```

Pure self play - 2m games, TD
```
  Ep 2000000/2000000 | eps=0.0010 | alpha=0.0010 | vs Random: 100.0%W/0.0%L/0.0%/1.0 | vs Naive: 99.8%W/0.0%L/0.2%/0.998 | 77410s
```

In theory the agent trained against the naive agent should perform better as it's trained to exploit the mistakes. Might expect 0.4% losses to account for risky play giving it an edge for more uncertain victories.

Observations on gameplay:
  * Pure self play policy values are basically 0 unless its got to a certain win position. This means it just moves around the board randomly essentially so as not to loose.
  * Pure naive play policy values are initialised between 0.6 and 0.8 for initial states. It seems it's incentivised to try in stay in good positions waiting for the naive agent to make a mistake. Would be interesting to compare average game length between the two agents.
