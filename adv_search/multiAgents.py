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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

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

        food_list = newFood.asList()
        close_food_dist = 0
        if food_list:
            close_food_dist = min(
                [manhattanDistance(newPos, food) for food in food_list]
            )

        close_ghost_dist = min(
            [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        )
        if min(newScaredTimes) > 0:
            close_ghost_dist = -close_ghost_dist * 3

        if action == "Stop":
            return 1 / close_food_dist

        return (
            (
                (15 / (close_food_dist + 1))
                + (80 / (successorGameState.getNumFood() + 1))
            )
            + close_ghost_dist / 8
            + successorGameState.getScore()
            - currentGameState.getScore()
        )


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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

        def min_max(gameState: GameState, agent_index=0, depth=0):
            if gameState.isWin() or gameState.isLose() or self.depth == depth:
                return [self.evaluationFunction(gameState)]

            if agent_index == gameState.getNumAgents() - 1:
                next_agent_index = self.index
                depth += 1
            else:
                next_agent_index = agent_index + 1

            best_action = None
            if agent_index == 0:
                v = float("-inf")
                for action in gameState.getLegalActions(agent_index):
                    new_minmax = min_max(
                        gameState.generateSuccessor(agent_index, action),
                        next_agent_index,
                        depth,
                    )[0]
                    if new_minmax > v:
                        v = new_minmax
                        best_action = action
                return v, best_action
            else:
                v = float("inf")
                for action in gameState.getLegalActions(agent_index):
                    new_minmax = min_max(
                        gameState.generateSuccessor(agent_index, action),
                        next_agent_index,
                        depth,
                    )[0]
                    if new_minmax < v:
                        v = new_minmax
                        best_action = action
                return v, best_action

        return min_max(gameState)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def max_value(gameState: GameState, agent_index, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            v = float("-inf")
            for max_action in gameState.getLegalActions(agent_index):
                v = max(
                    (
                        v,
                        min_value(
                            gameState.generateSuccessor(agent_index, max_action),
                            agent_index + 1,
                            depth,
                            alpha,
                            beta,
                        ),
                    )
                )
                if v > beta:
                    return v
                alpha = max((alpha, v))
            return v

        def min_value(gameState: GameState, agent_index, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            v = float("inf")
            for min_action in gameState.getLegalActions(agent_index):
                if agent_index < gameState.getNumAgents() - 1:
                    # There are still some ghosts to choose their moves, so increase agent index and call minValue again
                    v = min(
                        (
                            v,
                            min_value(
                                gameState.generateSuccessor(agent_index, min_action),
                                agent_index + 1,
                                depth,
                                alpha,
                                beta,
                            ),
                        )
                    )
                else:
                    # Depth is increased when it is MAX's turn
                    v = min(
                        (
                            v,
                            max_value(
                                gameState.generateSuccessor(agent_index, min_action),
                                0,
                                depth + 1,
                                alpha,
                                beta,
                            ),
                        )
                    )
                if v < alpha:
                    return v
                beta = min((beta, v))
            return v

        v = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        best_action = None
        for action in gameState.getLegalActions(0):
            cur_result = min_value(
                gameState.generateSuccessor(0, action), 1, 0, alpha, beta
            )
            if cur_result > v:
                v = cur_result
                best_action = action
                alpha = max((alpha, cur_result))
        return best_action


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

        def max_value(gameState: GameState, agent_index, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            successors = []
            for action in gameState.getLegalActions(0):
                successors.append(gameState.generateSuccessor(0, action))
            return max([exp_value(s, agent_index + 1, depth) for s in successors])

        def exp_value(gameState: GameState, agent_index, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)

            successors = []
            for action in gameState.getLegalActions(agent_index):
                successors.append(gameState.generateSuccessor(agent_index, action))
            if agent_index < gameState.getNumAgents() - 1:
                return sum(
                    [exp_value(s, agent_index + 1, depth) for s in successors]
                ) / len(successors)
            else:
                return sum([max_value(s, 0, depth + 1) for s in successors]) / len(
                    successors
                )

        actions = gameState.getLegalActions(0)
        best_action = "Stop"
        v = float("-inf")
        for a in actions:
            cur_result = exp_value(gameState.generateSuccessor(0, a), 1, 0)
            if cur_result > v:
                v = cur_result
                best_action = a
        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    value = 0
    food = currentGameState.getFood().asList()
    pacman_position = currentGameState.getPacmanPosition()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    time_scared_list = [
        ghost.scaredTimer for ghost in currentGameState.getGhostStates()
    ]
    for i in range(len(currentGameState.getGhostPositions())):
        if time_scared_list[i] > 0:
            value += -100 - (
                50
                * manhattanDistance(
                    pacman_position,
                    currentGameState.getGhostPositions()[i],
                )
            )
        elif (
            manhattanDistance(
                pacman_position,
                currentGameState.getGhostPositions()[i],
            )
            == 1
        ):
            return float("-inf")

    for f in food:
        value += -50 - (50 * manhattanDistance(pacman_position, f))

    for pellet in currentGameState.getCapsules():
        value += -150 - (
            50 * manhattanDistance(pacman_position, pellet)
        )

    return value + currentGameState.getScore() * 1000


# Abbreviation
better = betterEvaluationFunction
