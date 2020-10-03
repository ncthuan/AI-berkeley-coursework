###DFS, BFS, UCS, A*
DFS is implemented just as usual, with the use of stack for searching nodes
each node stores the position of the pacman and the path, which is the list of steps to take from the initial state

each of the rest follows the same pattern, and the major differences is the data structure being used
*BFS uses standard FIFO queue
*UCS uses priority queue with the cost of actions as priority value
*A* differs UCS in the priority value, which was calculated as the sum of cost of actions and heuristic function

###Corner problems
the state of 4 corners is represented by a list of 4 elements: [0,0,0,0]
0 denotes a corner is not visited by pacman
1 otherwise

###Corner heuristic
I used the largest manhattan distance from the pacman to 1 of the 4 corners