### DFS, BFS, UCS, A*
DFS is implemented just as usual, with the use of stack for backtracking.
Each node stores the position of the pacman and the path, which is the list of steps to take from the initial state.

Each of the rest follows the same pattern, and the major difference is the data structure being used.
* BFS uses standard FIFO queue
* UCS uses priority queue with the cost of actions as priority value
* A* differs from UCS in the priority value, which was calculated as the sum of cost of actions and heuristic function

One thing to note though, is the data structure I used for storing visited nodes. It would be much more efficient if I used a dict or 2d boolean array but for simplicity, I just used list.

### Q5 Corner problems
The state of 4 corners is represented by a list of 4 elements: [0,0,0,0]
0 denotes a corner is not visited by pacman
1 otherwise

### Q6 Corner heuristic
I used the largest manhattan distance from the pacman to 1 of the 4 corners

### Q7 Food heuristic
Similar to corner heuristic, my food heuristic first find the far most food using manhattan distance. And then calculate and return the maze distance from pacman to that faraway food. <br>
Further improvement can be done by designing a better heuristic.

### Q8 Suboptimal search

Overall result:
Question q1: 3/3
Question q2: 3/3
Question q3: 3/3
Question q4: 3/3
Question q5: 3/3
Question q6: 3/3
Question q7: 4/4
Question q8: 3/3
------------------
Total: 25/25