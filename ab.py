Search:
	echo "#!/bin/bash" > Search
	echo "python3 search.py \"\$$@\"" >> Search
	chmod u+x Search-bash-4.4$ 
-bash-4.4$ cat search.py
	
import kalah

ttable = {}
depths = {}

def depth_limited_search_strategy(depth, h):

	def fxn(pos):
		value, move = ab(pos, depth, h, -h.inf, h.inf)
		return move
	return fxn

def depth_unlimited_search_strategy(depth, h):
	def fxn(pos):
		value, move = ab(pos, depth, h, -h.inf, h.inf)
		return move
	return fxn

def ab(pos, depth, h, minimum, maximum):
	'''Returns the minimax value of the given position, with the given heuristic function
		applied at the given depth.

		pos -- a game position
		depth -- a nonnegative integer
		h -- a heuristic function that can be applied to pos and all its successors
		maximum -- greatest value in range
		minimum -- smallest value in range
	'''
	
	if pos.game_over() or depth == 0:
		if pos in ttable and depths[pos] >= depth:
			return ttable[pos]

		ttable[pos] = (h.evaluate(pos), None)
		depths[pos] = depth
		return ttable[pos]
		
	elif pos.next_player() == 0:
	# max player
		if pos in ttable and depths[pos] >= depth:
			return ttable[pos]

		best_value = minimum
		best_move = None
		moves = pos.legal_moves()
		for move in moves:
			child = pos.result(move)

			mm, _ = ab(child, depth - 1, h, best_value, maximum)
			if mm > best_value:
		
				if mm > maximum:
					ttable[pos] = (maximum, best_move)
					depths[pos] = depth
					return (maximum, best_move)
				
				best_value = mm
				best_move = move

		depths[pos] = depth
		return (best_value, best_move)

	else:
	# min player
		if pos in ttable and depths[pos] >= depth:
			return ttable[pos]

		best_value = maximum
		best_move = None
		moves = pos.legal_moves()
		for move in moves:
			child = pos.result(move)
			mm, _ = ab(child, depth - 1, h, minimum, best_value)
			if mm < best_value:
				
				
				best_value = mm
				best_move = move
		ttable[pos] = (best_value, best_move)
		depths[pos] = depth
		
		return (best_value, best_move)