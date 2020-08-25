import importlib
import numpy as np
import games.dao as dao
from open_spiel.python.bots import human
from open_spiel.python.bots import uniform_random
from open_spiel.python.algorithms import mcts
from absl import flags
from absl import app
import sys
import collections
from agents.naive import NaiveAgent

importlib.reload(dao)

_KNOWN_PLAYERS = [
    # A generic Monte Carlo Tree Search agent.
    "mcts",

    # A generic random agent.
    "random",

    # You'll be asked to provide the moves.
    "human",

    # Naive agent - one step look ahead
    "naive"
]

flags.DEFINE_enum("player1", "naive", _KNOWN_PLAYERS, "Who controls player 1.")
flags.DEFINE_enum("player2", "random", _KNOWN_PLAYERS, "Who controls player 2.")
flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("rollout_count", 1, "How many rollouts to do.")
flags.DEFINE_integer("max_simulations", 10, "How many simulations to run.")
flags.DEFINE_integer("num_games", 1, "How many games to play.")
flags.DEFINE_integer("seed", None, "Seed for the random number generator.")
flags.DEFINE_bool("solve", True, "Whether to use MCTS-Solver.")
flags.DEFINE_bool("quiet", False, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")
flags.DEFINE_integer("max_game_length", 100, "Maximum number of turns", lower_bound=1)

FLAGS = flags.FLAGS


def _opt_print(*args, **kwargs):
    if not FLAGS.quiet:
        print(*args, **kwargs)


def _init_bot(bot_type, game, player_id):
    """Initializes a bot by type."""
    rng = np.random.RandomState(FLAGS.seed)
    if bot_type == "mcts":
        evaluator = mcts.RandomRolloutEvaluator(FLAGS.rollout_count, rng)
        return mcts.MCTSBot(
            game,
            FLAGS.uct_c,
            FLAGS.max_simulations,
            evaluator,
            random_state=rng,
            solve=FLAGS.solve,
            verbose=FLAGS.verbose)
    if bot_type == "random":
        return uniform_random.UniformRandomBot(player_id, rng)
    if bot_type == "human":
        return human.HumanBot()
    if bot_type == "naive":
        return NaiveAgent(player_id, num_actions=game.num_distinct_actions())
    raise ValueError("Invalid bot type: %s" % bot_type)


def _get_action(state, action_str):
    for action in state.legal_actions():
        if action_str == state.action_to_string(state.current_player(), action):
            return action
    return None


def _play_game(game, bots):
    """Plays one game."""
    state = game.new_initial_state()

    _opt_print("Initial state:\n{}".format(state))

    history = []

    while not state.is_terminal():
        current_player = state.current_player()
        # The state can be three different types: chance node,
        # simultaneous node, or decision node
        if state.is_chance_node():
            # Chance node: sample an outcome
            outcomes = state.chance_outcomes()
            num_actions = len(outcomes)
            _opt_print("Chance node, got " + str(num_actions) + " outcomes")
            action_list, prob_list = zip(*outcomes)
            action = np.random.choice(action_list, p=prob_list)
            action_str = state.action_to_string(current_player, action)
            _opt_print("Sampled action: ", action_str)
        elif state.is_simultaneous_node():
            raise ValueError("Game cannot have simultaneous nodes.")
        else:
            # Decision node: sample action for the single current player
            bot = bots[current_player]
            action = bot.step(state)
            action_str = state.action_to_string(current_player, action)
            _opt_print("Player {} sampled action: {}".format(current_player,
                                                             action_str))

        for i, bot in enumerate(bots):
            if i != current_player:
                bot.inform_action(state, current_player, action)
        history.append(action_str)
        state.apply_action(action)

        _opt_print("Next state:\n{}".format(state))

    # Game is now done. Print return for each player
    returns = state.returns()
    print("Returns:", " ".join(map(str, returns)))

    for bot in bots:
        bot.restart()

    return returns, history


def main(argv):
    game = dao.DaoGame(FLAGS.max_game_length)
    if game.num_players() > 2:
        sys.exit("This game requires more players than the example can handle.")
    bots = [
        _init_bot(FLAGS.player1, game, 0),
        _init_bot(FLAGS.player2, game, 1),
    ]
    histories = collections.defaultdict(int)
    overall_returns = [0, 0]
    overall_wins = [0, 0]
    game_num = 0
    try:
        for game_num in range(FLAGS.num_games):
            returns, history = _play_game(game, bots)
            histories[" ".join(history)] += 1
            for i, v in enumerate(returns):
                overall_returns[i] += v
                if v > 0:
                    overall_wins[i] += 1
    except (KeyboardInterrupt, EOFError):
        game_num -= 1
        print("Caught a KeyboardInterrupt, stopping early.")
    print("Number of games played:", game_num + 1)
    print("Number of distinct games played:", len(histories))
    print("Players:", FLAGS.player1, FLAGS.player2)
    print("Overall wins", overall_wins)
    print("Overall returns", overall_returns)


# game = dao.DaoGame(max_game_length=100)
# state = game.new_initial_state()
# print(state)

# possible_actions = state.legal_actions()
# action_id = np.random.choice(possible_actions)
# state.apply_action(action_id)
# print(state)
# state.undo_action(action_id)
# print(state)

# print(state)
# print(state.rewards())

if __name__ == "__main__":
    app.run(main)
