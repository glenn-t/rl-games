import games.dao as dao
game = dao.DaoGame(max_game_length=10)
state = game.new_initial_state()
