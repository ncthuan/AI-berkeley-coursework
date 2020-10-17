# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

#from searchAgents import mazeDistance

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]


    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newCapsules = successorGameState.getCapsules() 
        inf = float('inf')

        # The state will have 5 features and its associated weights for evaluation
        food_weight = 1000              # ~ food_eaten
        food_dist_weight = -1           # ~ food_min_distance
        capsule_dist_weight = -5        # ~ capsule_min_distance
        ghost_dist2_weight = -2000      # ~ ghost_dist2 (if ghost is 2 distance away)
        ghost_dist1_weight = -10000     # ~ ghost_dist1 (if ghost is 1 distance away)
        scared_ghost_dist_weight = -20   # ~ scared_ghost_min_distance

        #1. food_eaten
        # if the next state have some food being eaten, then that's good
        food_eaten = (currentGameState.getNumFood() - successorGameState.getNumFood()) \
              +(len(currentGameState.getCapsules()) - len(successorGameState.getCapsules()))

        #2. food_min_distance
        # if next state pacman closer to food, then that's good
        food_min_distance = inf if newFood.asList() else 0
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            food_min_distance = min(food_min_distance, distance)

        #3. capsule_min_distance
        # if next state pacman closer to capsule(the power pellet), that's good as well
        capsule_min_distance = inf if newCapsules else 0
        for capsule in newCapsules:
            distance = manhattanDistance(newPos, capsule)
            capsule_min_distance = min(capsule_min_distance, distance)

        #4. ghost_dist2, ghost_dist1 and scared_ghost_min_distance
        # if pacman is very close to ghosts, that's bad
        # if pacman moving closer to scared (vulnerable) ghosts, that's good
        min_ghost_distance = inf
        scared_ghost_min_distance = inf
        for i, ghost in enumerate(newGhostStates):
            distance = manhattanDistance(newPos, ghost.getPosition())
            if newScaredTimes[i] > distance: # if the ghost scared & reachable
                scared_ghost_min_distance = min(scared_ghost_min_distance, distance)
            else:
                min_ghost_distance = min(min_ghost_distance, distance)
        
        ghost_dist2 = True if min_ghost_distance == 2 else False
        ghost_dist1 = True if min_ghost_distance <= 1 else False
        if scared_ghost_min_distance == inf:
            scared_ghost_min_distance = 0 

        return food_eaten*food_weight\
              +food_min_distance*food_dist_weight\
              +capsule_min_distance*capsule_dist_weight\
              +ghost_dist2*ghost_dist2_weight\
              +ghost_dist1*ghost_dist1_weight\
              +scared_ghost_min_distance*scared_ghost_dist_weight


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        legalMoves = gameState.getLegalActions(0)
        next_states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.minimax(next_state, current_depth=0, agent_id=1) for next_state in next_states]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) 

        return legalMoves[chosenIndex]

    def minimax(self, game_state, current_depth, agent_id):
        """
        Args:
          current_depth: the depth of the search tree that we are currently searching
          agent_id: 0 for pacman, >=1 for ghosts
        """
        # Pseudocode:
        # if the state is a terminal state: return the state's utility
        # if the next agent is MAX: return max-value(state)
        # if the next agent is MIN: return min-value(state)
        if current_depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if agent_id == 0:
            return self.max_value(game_state, current_depth)
        if agent_id >= 1:
            return self.min_value(game_state, current_depth, agent_id)

    def max_value(self, game_state, current_depth):
        # followed standard pseudocode
        v = float("-inf")
        for action in game_state.getLegalActions(0):
            v = max(v, self.minimax(game_state.generateSuccessor(0, action), current_depth, 1))
        return v

    def min_value(self, game_state, current_depth, agent_id):
        # slight modification to deal with multiple min agents:
        # if the current agent is the last min agent, move on the next depth, else continue with the next min agent
        v = float("inf")
        for action in game_state.getLegalActions(agent_id):
            if agent_id == game_state.getNumAgents()-1:
                v = min(v, self.minimax(game_state.generateSuccessor(agent_id, action), current_depth+1, 0))
            else:
                v = min(v, self.minimax(game_state.generateSuccessor(agent_id, action), current_depth, agent_id+1))
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        alpha = float("-inf")
        beta = float("inf")

        legalMoves = gameState.getLegalActions(0)
        best_score = float("-inf")
        best_action = None
        for action in legalMoves:
            next_state = gameState.generateSuccessor(0, action)
            score = self.minimax_ab(next_state, 0, 1, alpha, beta)
            if best_score < score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score) # modification to adapt the pruning scheme

        return best_action

    def minimax_ab(self, game_state, current_depth, agent_id, alpha, beta):
        if current_depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if agent_id == 0:
            return self.max_value(game_state, current_depth, alpha, beta)
        if agent_id >= 1:
            return self.min_value(game_state, current_depth, agent_id, alpha, beta)

    def max_value(self, game_state, current_depth, alpha, beta):
        v = float("-inf")
        for action in game_state.getLegalActions(0):
            v = max(v, self.minimax_ab(game_state.generateSuccessor(0, action), current_depth, 1, alpha, beta))
            if v > beta: return v
            alpha = max(alpha, v)
        return v

    def min_value(self, game_state, current_depth, agent_id, alpha, beta):
        v = float("inf")
        for action in game_state.getLegalActions(agent_id):
            if agent_id == game_state.getNumAgents()-1:
                v = min(v, self.minimax_ab(game_state.generateSuccessor(agent_id, action), current_depth+1, 0, alpha, beta))
            else:
                v = min(v, self.minimax_ab(game_state.generateSuccessor(agent_id, action), current_depth, agent_id+1, alpha, beta))
            if v < alpha: return v
            beta = min(beta, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions(0)
        next_states = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [self.expectimax(next_state, 0, 1) for next_state in next_states]

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) 

        return legalMoves[chosenIndex]

    def expectimax(self, game_state, current_depth, agent_id):
        # Pseudocode:
        # if the state is a terminal state: return the state's utility
        # if the next agent is MAX: return max-value(state)
        # if the next agent is EXP: return exp-value(state)
        if current_depth == self.depth or game_state.isWin() or game_state.isLose():
            return self.evaluationFunction(game_state)
        if agent_id == 0:
            return self.max_value(game_state, current_depth)
        if agent_id >= 1:
            return self.exp_value(game_state, current_depth, agent_id)

    def max_value(self, game_state, current_depth):
        v = float("-inf")
        for action in game_state.getLegalActions(0):
            v = max(v, self.expectimax(game_state.generateSuccessor(0, action), current_depth, 1))
        return v

    def exp_value(self, game_state, current_depth, agent_id):
        v = 0
        successor_state_actions = game_state.getLegalActions(agent_id)
        for action in successor_state_actions:
            p = 1.0/len(successor_state_actions)
            if agent_id == game_state.getNumAgents()-1:
                v += p*self.expectimax(game_state.generateSuccessor(agent_id, action), current_depth+1, 0)
            else:
                v += p*self.expectimax(game_state.generateSuccessor(agent_id, action), current_depth, agent_id+1)
        return v



def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
        basically reuse the evaluation function from ReflexAgent
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules() 
    inf = float('inf')

    # The state will have 5 features and its associated weights for evaluation
    food_weight = -1000             # ~ num_food
    food_dist_weight = -1           # ~ food_min_distance
    capsule_dist_weight = -20       # ~ capsule_min_distance
    ghost_dist2_weight = -2000      # ~ ghost_dist2 (if ghost is 2 distance away)
    ghost_dist1_weight = -10000     # ~ ghost_dist1 (if ghost is 1 distance away)
    scared_ghost_dist_weight = -50  # ~ scared_ghost_min_distance

    #1. num_food
    # if the next state have less food, then that's good
    num_food = currentGameState.getNumFood() + len(currentGameState.getCapsules())

    #2. food_min_distance
    # if next state pacman closer to food, then that's good
    food_min_distance = inf if newFood.asList() else 0
    for food in newFood.asList():
        distance = manhattanDistance(newPos, food)
        food_min_distance = min(food_min_distance, distance)

    #3. capsule_min_distance
    # if next state pacman closer to capsule(the power pellet), that's good as well
    capsule_min_distance = inf if newCapsules else 0
    for capsule in newCapsules:
        distance = manhattanDistance(newPos, capsule)
        capsule_min_distance = min(capsule_min_distance, distance)

    #4. ghost_dist2, ghost_dist1 and scared_ghost_min_distance
    # if pacman is very close to ghosts, that's bad
    # if pacman moving closer to scared (vunerable) ghosts, that's good
    min_ghost_distance = inf
    scared_ghost_min_distance = inf
    for i, ghost in enumerate(newGhostStates):
        distance = manhattanDistance(newPos, ghost.getPosition())
        if newScaredTimes[i] > distance: # if the ghost scared & reachable
            scared_ghost_min_distance = min(scared_ghost_min_distance, distance)
        else:
            min_ghost_distance = min(min_ghost_distance, distance)
    
    ghost_dist2 = True if min_ghost_distance == 2 else False
    ghost_dist1 = True if min_ghost_distance <= 1 else False
    if scared_ghost_min_distance == inf:
        scared_ghost_min_distance = 0 

    score = currentGameState.getScore()

    return score + num_food*food_weight\
          +food_min_distance*food_dist_weight\
          +capsule_min_distance*capsule_dist_weight\
          +ghost_dist2*ghost_dist2_weight\
          +ghost_dist1*ghost_dist1_weight\
          +scared_ghost_min_distance*scared_ghost_dist_weight


# Abbreviation
better = betterEvaluationFunction
