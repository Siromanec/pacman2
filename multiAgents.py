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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(scores)
        # print(legalMoves[chosenIndex], chosenIndex)
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        def dist(xy1, xy2):
            return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
        def search_min(xy1, collection):
            return min(collection, key=lambda xy2: dist(xy1, xy2))
        min_food_dist = 100000
        if len(newFood.asList()) == 0:
            min_food_dist = 0
        elif action != "Stop":
            min_food_dist = dist(search_min(newPos, newFood.asList()), newPos)
        min_food_dist = (1/(min_food_dist) if min_food_dist != 0 else 1.1)
        total_dist = sum(map(lambda x: dist(x[0].getPosition(), newPos) if x[1] == 0 else 0, zip(newGhostStates, newScaredTimes)))
        ghost_direction = newGhostStates[0].getDirection()
        total_dist = -float("inf") if total_dist == 0 else 0
        score = successorGameState.getScore() + total_dist + min_food_dist
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax_search(gameState):

            value, move = max_value(gameState, self.depth)
            return move

        def max_value(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None
            v = float("-inf")
            move = None
            for a in gameState.getLegalActions(0):
                v2, a2 = min_value(gameState.generateSuccessor(0, a), depth-1, 1)
                if v2 > v:
                    v, move = v2, a
            return v, move

        def min_value(gameState,  depth, playerIndex):
            if gameState.isLose() or gameState.isWin() or depth == -1:
                return self.evaluationFunction(gameState), None
            v = float("inf")
            move = None
            for a in gameState.getLegalActions(playerIndex):
                if (playerIndex+1)%(gameState.getNumAgents()) == 0:
                    v2, a2 = max_value(gameState.generateSuccessor(playerIndex, a),  depth)
                else:
                    v2, a2 = min_value(gameState.generateSuccessor(playerIndex, a),  depth, playerIndex + 1)
                if v2 < v:
                    v, move = v2, a
            return v, move
        # there is a need to create min value function for each ghost and chain it together
        # only pacman function should decrement the depth
        # to chain the functions i could use chain of responsibility... embedded into the functions
        #
        return minimax_search(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax_search(gameState):
            value, move = max_value(gameState, self.depth, float("-inf"), float("inf"))
            return move

        def max_value(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None
            v = float("-inf")
            move = None
            for a in gameState.getLegalActions(0):
                v2, a2 = min_value(gameState.generateSuccessor(0, a), depth-1, 1, alpha, beta)
                if v2 > v:
                    v, move = v2, a
                    alpha = max((alpha, v))
                if v > beta: return v, move
            return v, move

        def min_value(gameState,  depth, playerIndex, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == -1:
                return self.evaluationFunction(gameState), None
            v = float("inf")
            move = None
            for a in gameState.getLegalActions(playerIndex):
                if (playerIndex+1)%(gameState.getNumAgents()) == 0:
                    v2, a2 = max_value(gameState.generateSuccessor(playerIndex, a),  depth, alpha, beta)
                else:
                    v2, a2 = min_value(gameState.generateSuccessor(playerIndex, a),  depth, playerIndex + 1, alpha, beta)
                if v2 < v:
                    v, move = v2, a
                    beta = min((beta, v))
                if v < alpha: return v, move
            return v, move
        return minimax_search(gameState)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minimax_search(gameState):

            value, move = max_value(gameState, self.depth)
            return move

        def max_value(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None
            v = float("-inf")
            move = None
            for a in gameState.getLegalActions(0):
                v2, a2 = min_value(gameState.generateSuccessor(0, a), depth-1, 1)
                if v2 > v:
                    v, move = v2, a
            return v, move

        def min_value(gameState,  depth, playerIndex):
            if gameState.isLose() or gameState.isWin() or depth == -1:
                return self.evaluationFunction(gameState), None
            v = float("inf")
            move = None
            actions = gameState.getLegalActions(playerIndex)
            u_sum = 0
            for a in actions:
                if (playerIndex+1)%(gameState.getNumAgents()) == 0:
                    v2, a2 = max_value(gameState.generateSuccessor(playerIndex, a),  depth)
                else:
                    v2, a2 = min_value(gameState.generateSuccessor(playerIndex, a),  depth, playerIndex + 1)
                # v2*=(1/len(actions))
                u_sum += v2
                if v2 < v:
                    v, move = v2, a
            
            return u_sum/len(actions), move
        return minimax_search(gameState)

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    MAX_VAL = 1000000
    def dist(xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    def search_min(xy1, collection):
        return min(collection, key=lambda xy2: dist(xy1, xy2))
    if currentGameState.isWin():

        return MAX_VAL
    if currentGameState.isLose():
        return -MAX_VAL
    # actions = currentGameState.getLegalActions(0)
    # print(actions)
    score = 0
    # for action in actions:
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # "*** YOUR CODE HERE ***"

    min_food_dist = MAX_VAL
    min_cap_dist = 0
    if currentGameState.getNumFood() == 0:
        min_food_dist = 0
    else:
        min_food_dist = dist(search_min(newPos, newFood.asList()), newPos)
    min_food_dist = (1/(min_food_dist) if min_food_dist != 0 else 1.1)

    total_dist = sum(map(lambda x: dist(x[0].getPosition(), newPos) if x[1] == 0 else 0, zip(newGhostStates, newScaredTimes))) #/ currentGameState.getNumAgents()
    
    # ghost_direction = newGhostStates[0].getDirection()
    # if total_dist > 5:
    #     capsules = currentGameState.getCapsules()
    #     if len(capsules) != 0:
    #         min_cap_dist = dist(search_min(newPos, capsules), newPos)
    #     else:
    #         min_cap_dist = 0
    #     min_cap_dist = (1/(min_cap_dist) if min_cap_dist != 0 else 1)

    total_dist = -MAX_VAL if total_dist <= 1 else 0

    score += currentGameState.getScore() + total_dist + min_food_dist + min_cap_dist

    if betterEvaluationFunction2(currentGameState.generatePacmanSuccessor("Stop")) == score:
        score -= MAX_VAL

    return score
    # return currentGameState.getScore()
def betterEvaluationFunction2(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    MAX_VAL = 1000000
    def dist(xy1, xy2):
        return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    def search_min(xy1, collection):
        return min(collection, key=lambda xy2: dist(xy1, xy2))
    if currentGameState.isWin():

        return MAX_VAL
    if currentGameState.isLose():
        return -MAX_VAL
    # actions = currentGameState.getLegalActions(0)
    # print(actions)
    score = 0
    # for action in actions:
    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    # "*** YOUR CODE HERE ***"

    min_food_dist = MAX_VAL
    min_cap_dist = 0
    if currentGameState.getNumFood() == 0:
        min_food_dist = 0
    else:
        min_food_dist = dist(search_min(newPos, newFood.asList()), newPos)
    min_food_dist = (1/(min_food_dist) if min_food_dist != 0 else 1.1)

    total_dist = sum(map(lambda x: dist(x[0].getPosition(), newPos) if x[1] == 0 else 0, zip(newGhostStates, newScaredTimes))) #/ currentGameState.getNumAgents()
    
    # ghost_direction = newGhostStates[0].getDirection()
    # if total_dist > 5:
    #     capsules = currentGameState.getCapsules()
    #     if len(capsules) != 0:
    #         min_cap_dist = dist(search_min(newPos, capsules), newPos)
    #     else:
    #         min_cap_dist = 0
    #     min_cap_dist = (1/(min_cap_dist) if min_cap_dist != 0 else 1)

    total_dist = -MAX_VAL if total_dist <= 1 else 0

    score += currentGameState.getScore() + total_dist + min_food_dist + min_cap_dist
    return score
# Abbreviation
better = betterEvaluationFunction
