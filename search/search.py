# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    stack = util.Stack() #(pos_xy, path)  --((5,5), [s,w,e])
    visited = []
    path = []

    # Initiate at the start state
    stack.push((problem.getStartState(), []))

    while(True):
        # The stack empty means every node is searched and no solution found
        if stack.isEmpty():
            return []

        # Traverse to the current state, get pos and path
        pos_xy, path = stack.pop()

        # If found a solution to goal, return the path to it
        if problem.isGoalState(pos_xy):
            return path

        if pos_xy not in visited:         
            visited.append(pos_xy)
            # Get successors of current state and push them to the stack   
            successors = problem.getSuccessors(pos_xy)
            for next_state, action,_ in successors:
                if next_state not in visited:
                    new_path = path + [action]
                    stack.push((next_state, new_path))
    

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    queue = util.Queue()
    visited = []
    path = []

    # Initiate at the start state
    queue.push((problem.getStartState(), []))

    while(True):
        # The stack empty means every node is searched and no solution found
        if queue.isEmpty():
            return []

        # Traverse to the current state, get pos and path
        pos_xy, path = queue.pop()

        # If found a solution to goal, return the path to it
        if problem.isGoalState(pos_xy):
            return path

        if pos_xy not in visited:         
            visited.append(pos_xy)
            # Get successors of current state and push them to the queue   
            successors = problem.getSuccessors(pos_xy)
            for next_state, action,_ in successors:
                if next_state not in visited:
                    new_path = path + [action]
                    queue.push((next_state, new_path))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    pqueue = util.PriorityQueue() #(pos, path, cost), priority_val
    visited = []
    path = []

    # Initiate at the start state, with the cheapest priority
    pqueue.push((problem.getStartState(), [], 0), 0)

    while(True):
        # The stack empty means every node is searched and no solution found
        if pqueue.isEmpty():
            return []

        # Traverse to the current state, get pos and path
        pos_xy, path, cost = pqueue.pop()

        # If found a solution to goal, return the path to it
        if problem.isGoalState(pos_xy):
            return path

        if pos_xy not in visited:         
            visited.append(pos_xy)
            # Get successors of current state and push them to the queue   
            successors = problem.getSuccessors(pos_xy)
            for next_state, action, action_cost in successors:
                if next_state not in visited:
                    new_path = path + [action]
                    next_state_cost =  cost + action_cost
                    pqueue.push((next_state, new_path, next_state_cost), next_state_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def f(problem, state, heuristic):
    return problem.getCostOfActions(state[1]) + heuristic(state[0],problem)


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    pqueue = util.PriorityQueue() #(pos, path, cost), priority_val
    visited = []
    path = []

    # Initiate at the start state, with the cheapest priority
    pqueue.push((problem.getStartState(), [], 0), 0)

    while(True):
        # The stack empty means every node is searched and no solution found
        if pqueue.isEmpty():
            return []

        # Traverse to the current state, get pos and path
        pos_xy, path, cost = pqueue.pop()

        # If found a solution to goal, return the path to it
        if problem.isGoalState(pos_xy):
            return path

        if pos_xy not in visited:         
            visited.append(pos_xy)
            # Get successors of current state and push them to the queue   
            successors = problem.getSuccessors(pos_xy)
            for next_state, action, action_cost in successors:
                if next_state not in visited:
                    new_path = path + [action]
                    next_state_cost =  cost + action_cost
                    # f(n) = g(n) + h(n)
                    f_cost = next_state_cost + heuristic(next_state, problem)
                    pqueue.push((next_state, new_path, next_state_cost), f_cost)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
