import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    #   cpy           - whether the player is a copy (when playing itself)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "Ai")
        self.depth_limit = 3
        self.anthillCoords = None
        self.tunnelCoords = None
        self.myFoodCoords = None
        self.maxTunnelDist = 0
        self.maxFoodDist = 0
        self.heuristicRuns = 0
        self.prunesZero = 0
        self.prunesOne = 0
        self.prunesTwo = 0
        self.forwardPruneLength = 5


    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]

    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        self.heuristicRuns = 0
        self.prunesZero = 0
        self.prunesOne = 0
        self.prunesTwo = 0
        maxPlayer = currentState.whoseTurn
        root = {"move":None, "state":currentState, "value":0, "parent":None, "depth":0}
        move = self.alphabeta(root, 0, -1, 1, maxPlayer)
        print("huer-runs: " + str(self.heuristicRuns))
        print("0-prunes: " + str(self.prunesZero))
        print("1-prunes: " + str(self.prunesOne))
        print("2-prunes: " + str(self.prunesTwo))
        return move

    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # registerWin
    #
    # This agent doens't learn
    # however we need to reset global variables
    #
    def registerWin(self, hasWon):
        #reset global variables
        self.depth_limit = 3
        self.anthillCoords = None
        self.tunnelCoords = None
        self.myFoodCoords = None
        self.maxTunnelDist = 0
        self.maxFoodDist = 0
        self.forwardPruneLength = 5
        pass

    ##
    #evaluateState
    #
    # This agent evaluates the state and returns a double between -1.0 and 1.0
    #
    def evaluateState(self, gs, me):
        self.heuristicRuns += 1
        if gs.whoseTurn == me:
            myInv = getCurrPlayerInventory(gs)
            theirInv = getEnemyInv(self, gs)
        else:
            myInv = getEnemyInv(self, gs)
            theirInv = getCurrPlayerInventory(gs)
        myQueen = getAntList(gs, me, (QUEEN,))[0]
        # enemyAntsThreat = getAntList(gs, 1-me, (DRONE, R_SOLDIER))
        myWorkers = getAntList(gs, me, (WORKER,))
        myAnts = getAntList(gs, me, (WORKER,QUEEN,WORKER,DRONE,SOLDIER,R_SOLDIER))
        if len(getAntList(gs, 1 - me, (QUEEN,))) == 0:
            return 1
        enemyQueen = getAntList(gs, 1 - me, (QUEEN,))[0]
        # enemyWorkers = getAntList(gs, 1 - me, (WORKER,))

        ###################################
        #### INIT ######################
        ###################################

        # for ant in enemyAntsThreat:
        #     if ant.coords[1] > 3:
        #         enemyAntsThreat.remove(ant)

        #do this once
        if self.maxTunnelDist == 0:
            self.tunnelCoords = getConstrList(gs, me, (TUNNEL,))[0].coords
            self.anthillCoords = getConstrList(gs, me, (ANTHILL,))[0].coords
            foods = getConstrList(gs, None, (FOOD,))
            #find the food closest to the tunnel
            bestDistSoFar = 1000 #i.e., infinity
            for food in foods:
                dist = stepsToReach(gs, self.tunnelCoords, food.coords)
                if (dist < bestDistSoFar):
                    self.myFoodCoords = food.coords
                    bestDistSoFar = dist

            for i in range(0,10):
                for j in range(0,10):
                    tunnelDist = approxDist((i,j), self.tunnelCoords)
                    foodDist = approxDist((i,j), self.myFoodCoords)
                    if (tunnelDist > self.maxTunnelDist):
                        self.maxTunnelDist = tunnelDist
                    if (foodDist > self.maxFoodDist):
                        self.maxFoodDist = foodDist

        ###################################
        #### RETURN BAD ######################
        ###################################

        #if we have more than two workers, throw out this option
        if len(myWorkers) > 2:
            #print(-1)
            return -1.0

        if len(getAntList(gs, me, (R_SOLDIER, DRONE))) > 0:
            #print(-1)
            return -1

        #remove nodes where an ant is sitting idle on the hill
        if myQueen.coords == self.anthillCoords:
            #print(-1)
            return -1.0




        ###################################
        #### ATTACK ######################
        ###################################
        myAttackers = getAntList(gs, me, (SOLDIER,))
        if len(myAttackers) > 0:
            attackScore = 0
            maxAttackScore = 0
            for ant in myAttackers:
                # if len(enemyWorkers) > 0:
                #     attackScore += 20 - approxDist(ant.coords, enemyWorkers[0].coords)
                # else:
                attackScore += 20 - approxDist(ant.coords, enemyQueen.coords)
                maxAttackScore += 19
            attackScore = attackScore / maxAttackScore
        else:
            attackScore = 0




        ###################################
        #### ANT NUM ######################
        ###################################

        #calculate ant score
        myAntScore = 0
        for ant in myAnts:
            if ant.type == WORKER:
                myAntScore += 1
            elif ant.type == SOLDIER:
                myAntScore += 4
            elif ant.type == DRONE:
                myAntScore += 2
            elif ant.type == R_SOLDIER:
                myAntScore += 3

        theirAnts = theirInv.ants
        theirAntScore = 0
        for ant in theirAnts:
            if ant.type == WORKER:
                theirAntScore += 1
            elif ant.type == SOLDIER:
                theirAntScore += 4
            elif ant.type == DRONE:
                theirAntScore += 2
            elif ant.type == R_SOLDIER:
                theirAntScore += 3
        antDiff = (myAntScore - theirAntScore) / max(myAntScore, theirAntScore)




        ###################################
        #### FOOD DIST ######################
        ###################################
        myCarryScore = 0
        depositScore = 0
        # Non carrying workers are close to food and carrying workers are close to tunnel
        collectScore = 0
        for worker in myWorkers:
            if worker.carrying:
                myCarryScore += 1
                depositScore += 1 - (approxDist(worker.coords, self.tunnelCoords) / self.maxTunnelDist)
            else:
                collectScore += 1 - (approxDist(worker.coords, self.myFoodCoords) / self.maxFoodDist)

        foodDistScore = (depositScore + collectScore) / 2



        ###################################
        #### FOOD SCORE ######################
        ###################################
        # D. How much food each player has
        myfoodScore = (myInv.foodCount) / 11




        ###################################
        #### CARRY ######################
        ###################################
        # E. How much food the worker ants are carrying
        if len(myWorkers) != 0:
            myCarryScore = myCarryScore / len(myWorkers)
        else:
            myCarryScore = 0




        ##############################
        ###### FINAL RETURN ##########
        ##############################


        output = (20*myfoodScore + 20*antDiff + myCarryScore + foodDistScore + 10*attackScore) / 53
        #print(output)
        return output

    ##
    # expandNode
    #
    # This function takes a node (dictionary) as input finds all the legal moves from that state
    # and creates a list of new node with states resulting from each of those nodes and returns that list
    #
    def expandNode(self, node, maxPlayer):
        moves = listAllLegalMoves(node["state"])
        states = []
        fullNodeList = []
        forwardPrunedNodes = []
        for move in moves:
            newNode = {"move":move}
            newNode["state"] = getNextStateAdversarial(node["state"], newNode["move"])
            newNode["value"] = self.evaluateState(newNode["state"], maxPlayer)
            newNode["parent"] = node
            newNode["depth"] = node["depth"]+1
            states.append(newNode)
        if node["state"].whoseTurn == maxPlayer:
            fullNodeList = sorted(states, key=lambda k: k["value"], reverse=True)
        else:
            fullNodeList = sorted(states, key=lambda k: k["value"])
        if len(fullNodeList) > self.forwardPruneLength:
            for i in range(0, self.forwardPruneLength):
                forwardPrunedNodes.append(fullNodeList[i])
        else:
            forwardPrunedNodes = fullNodeList

        return forwardPrunedNodes

    ##
    # evalListNodes
    #
    # This function takes a list of nodes and takes the average of the values accossiated with
    # each and returns that average value
    #
    def evalListNodes(self, nodes, maxPlayer):
        #if parent maxPlayer
        if nodes[0]["parent"]["state"].whoseTurn == maxPlayer:
            #print("max")
            maxNode = max(nodes, key=lambda k: k["value"])
            return maxNode["value"]
        else:
            #print("min")
            minNode = min(nodes, key=lambda k: k["value"])
            return minNode["value"]


    ##
    # bfs
    #
    # This function preforms breadth first search
    # It takes a node and the depth
    # The nodes are expanded untill the depth limit is reached
    # Then the values of the nodes are averaged and propigated up
    # The move from the node with the best value at depth 0 is returned as the best move to make
    #
    def alphabeta(self, node, depth, alpha, beta, maxPlayer):
        #it is depth + 1 since we just expanded the node and are
        #now evaluating nodes at depth + 1
        if depth < self.depth_limit:
            newNodes = self.expandNode(node, maxPlayer)
            value = 0
            if newNodes[0]["parent"]["state"].whoseTurn == maxPlayer:
                value = -1
                for n in newNodes:
                    value = max(value, self.alphabeta(n, depth+1, alpha, beta, maxPlayer))
                    alpha = max(alpha, value)
                    if alpha+.2 >= beta:
                        if depth == 0:
                            print(alpha)
                            self.prunesZero += 1
                        elif depth == 1:
                            self.prunesOne += 1
                        elif depth == 2:
                            self.prunesTwo += 1
                        break
            else:
                value = 1
                for n in newNodes:
                    value = min(value, self.alphabeta(n, depth+1, alpha, beta, maxPlayer))
                    beta = min(beta, value)
                    if alpha+.2 >= beta:
                        if depth == 0:
                            self.prunesZero += 1
                        elif depth == 1:
                            self.prunesOne += 1
                        elif depth == 2:
                            self.prunesTwo += 1
                        break
            if depth == 0:
                maxNode = max(newNodes, key=lambda k: k["value"])
                #print(newNodes.index(maxNode))
                return maxNode["move"]
            else:
                node["value"] = value
                return value

        else:
            # #else find the values of each node
            # for n in newNodes:
            #     n["value"] = self.evaluateState(n["state"], maxPlayer)
            # evaluation = self.evalListNodes(newNodes, maxPlayer)
            # node["value"] = self.evaluateState(node["state"], maxPlayer)
            return node["value"]
