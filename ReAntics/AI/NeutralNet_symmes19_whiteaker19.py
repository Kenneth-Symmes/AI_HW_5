  # -*- coding: latin-1 -*-
import random
import sys
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import addCoords
from AIPlayerUtils import *
import math

class Node:
    def __init__(self,initState,parentNode,moveToGetHere,utility):
        self.parent=parentNode
        self.move=moveToGetHere
        self.children=[]
        self.util=utility
        self.state=initState

    def addChild(self,child):
        self.children.append(child)
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
        super(AIPlayer, self).__init__(inputPlayerId, "NeutralNet1")
        #the coordinates of the agent's food and tunnel will be stored in these
        #variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None
        self.myAnthill=None
        self.workersWanted=3
        self.enemyAnthill=None
        self.enemyTunnel=None
        self.depth=2
        self.learning_rate = 100
        self.num_inputs = 26
        self.hidden_nodes = int((self.num_inputs) * (2/3))
        self.ticks = 0
        self.inputs=[]
        for i in range(0,self.num_inputs):
            self.inputs.append(0)

        self.weights=[]
        for i in range(0,self.hidden_nodes):
            temp=[]
            for j in range(0,self.num_inputs+1):
                temp.append(random.uniform(-1,1))
            self.weights.append(temp)

        temp=[]
        for i in range(0,self.hidden_nodes + 1):
            temp.append(random.uniform(-1,1))
        self.weights.append(temp)


    ##
    #getPlacement
    #
    # The agent uses a hardcoded arrangement for phase 1 to provide maximum
    # protection to the queen.  Enemy food is placed randomly.
    #
    def getPlacement(self, currentState):
        self.myFood = None
        self.myTunnel = None
        self.myFood=None
        self.enemyAnthill=None
        self.enemyTunnel=None
        if currentState.phase == SETUP_PHASE_1:
            options=[]
            for i in range(0,10):
                for j in range(0,4):
                    options.append((i,j))
            chosen=[]
            for i in range(0,11):
                pick=random.randint(0,len(options)-1)
                #print(str(len(options))+", "+str(pick))
                chosen.append(options[pick])
                del options[pick]
            self.myAnthill=chosen[0]
            self.myTunnel=chosen[1]
            return chosen
        elif currentState.phase == SETUP_PHASE_2:
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
            return None  #should never happen

    ##
    #getMove
    #
    # This agent simply gathers food as fast as it can with its worker.  It
    # never attacks and never builds more ants.  The queen is never moved.
    #
    ##
    def getMove(self, currentState):
        me=currentState.whoseTurn
        if(self.myFood==None):
            self.myFood=getCurrPlayerFood(self,currentState)
            self.myFood[0]=self.myFood[0].coords
            self.myFood[1]=self.myFood[1].coords
            self.myTunnel=(getConstrList(currentState, me, (TUNNEL,))[0]).coords
            self.myAnthill=(getConstrList(currentState, me, (ANTHILL,))[0]).coords
            self.enemyAnthill=(getConstrList(currentState, 1-me, (ANTHILL,))[0]).coords
            self.enemyTunnel=(getConstrList(currentState, 1-me, (TUNNEL,))[0]).coords
        startUtil=self.getUtil(currentState)
        self.getNetUtil(currentState,startUtil)
        #print(str(startUtil))
        root=Node(currentState,None,None,startUtil)
        ourNode=root
        self.grow(root,0) #makes tree of possible states
        highest=[Move(END,None,None),startUtil]
        self.findHighest(root,highest) #finds best end state
        #print(str(highest[1]))
        return highest[0]

    def getNetUtil(self,currentState,initUtil):
        self.setInputs(currentState)
        self.backpropigation(initUtil)

    def setInputs(self,cur):
        me=cur.whoseTurn
        myInv=-1
        eInv=-1
        workers=getAntList(cur, me, (WORKER,))
        enemyQueen=getAntList(cur,1-me,(QUEEN,))
        enemyWorkers=getAntList(cur,1-me,(WORKER,))
        myQueen=getAntList(cur,me,(QUEEN,))[0]
        myArmy=getAntList(cur, me, (SOLDIER,R_SOLDIER))
        myArmyS=getAntList(cur, me, (SOLDIER,))
        myArmyR=getAntList(cur, me, (R_SOLDIER,))
        enemyArmy=getAntList(cur,1-me,(DRONE,SOLDIER,R_SOLDIER))

        if(cur.whoseTurn==me):
            myInv=getCurrPlayerInventory(cur)
            eInv=getEnemyInv(self,cur)
        else:
            eInv=getCurrPlayerInventory(cur)
            myInv=getEnemyInv(self,cur)

        self.inputs[0]=myInv.foodCount/11; #ourFoodCount 0-1
        self.inputs[1]=eInv.foodCount/11; #enemyFoodCount 0-1
        self.inputs[2]=myQueen.coords[1]/3; #ourQueenPos, 0,.33,.67, or 1
        self.inputs[3]=len(enemyQueen) #0 or 1 enemyQueen
        if(len(enemyQueen)!=0):
            self.inputs[4]=enemyQueen[0].health/10 #enemyQueenHealth 0-1
        else:
            self.inputs[4]=0
        self.inputs[5]=len(enemyWorkers)/100 #number of enemyWorkers 0-1
        self.inputs[6]=len(myArmyS)/100 #number of Soldiers
        self.inputs[7]=len(myArmyR)/100 #number of RSoldiers
        self.inputs[8]=abs(len(myArmyS)-len(myArmyR))/100 #Diff of R and S soldiers
        #these next 5 are soldier stats
        self.inputs[9]=0 #health
        self.inputs[18]=0 #attack
        self.inputs[19]=0 #movement Ability
        self.inputs[20]=0 #range
        self.inputs[21]=0 #cost

        self.inputs[11]=0 #1 if soldier is on enemyAnthill
        self.inputs[12]=0 #Total soldier distance to enemy Anthill (1 if none of this type)
        self.inputs[13]=0 #Total ranged soldier distance to enemy Tunnel (1 if none of this type)
        if(len(myArmyS)==0):
            self.inputs[12]=1
        if(len(myArmyR)==0):
            self.inputs[13]=1
        for w in myArmy:
            self.inputs[9]+=w.health/500
            self.inputs[18]+=UNIT_STATS[w.type][2]/200
            self.inputs[19]+=UNIT_STATS[w.type][0]/200
            self.inputs[20]+=UNIT_STATS[w.type][3]/200
            self.inputs[21]+=UNIT_STATS[w.type][4]/200

            if(w.coords==self.enemyAnthill):
                self.inputs[11]=1;
            if(w.type==SOLDIER):
                self.inputs[12]+=approxDist(w.coords,self.enemyAnthill)/1000
            elif(w.type==R_SOLDIER):
                self.inputs[13]+=approxDist(w.coords,self.enemyTunnel)/1000

        self.inputs[10]=0 #same as 9,18,19,20,21 but for enemy
        self.inputs[22]=0
        self.inputs[23]=0
        self.inputs[24]=0
        self.inputs[25]=0
        for w in enemyArmy:
            self.inputs[10]+=w.health/500
            self.inputs[22]+=UNIT_STATS[w.type][2]/200
            self.inputs[23]+=UNIT_STATS[w.type][0]/200
            self.inputs[24]+=UNIT_STATS[w.type][3]/200
            self.inputs[25]+=UNIT_STATS[w.type][4]/200

        self.inputs[14]=0 #number of workers carrying Food
        self.inputs[15]=0 #total worker distance to their destinations
        for w in workers:
            if(w.carrying):
                self.inputs[14]+=0.01
                close=approxDist(w.coords,self.myAnthill)
                close2=approxDist(w.coords,self.myTunnel)
                if(close2<close):
                    close=close2
                self.inputs[15]+=close/200
            else:
                close=approxDist(w.coords,self.myFood[0])
                close2=approxDist(w.coords,self.myFood[1])
                if(close2<close):
                    close=close2
                self.inputs[15]+=close/200
        self.inputs[16]=0 #number of ants covering things up
        if(getAntAt(cur,self.myAnthill)!=None):
            self.inputs[16]+=.25
        if(getAntAt(cur,self.myTunnel)!=None):
            self.inputs[16]+=.25
        if(getAntAt(cur,self.myFood[0])!=None):
            self.inputs[16]+=.25
        if(getAntAt(cur,self.myFood[1])!=None):
            self.inputs[16]+=.25
        self.inputs[17]=(self.workersWanted-abs(len(workers)-self.workersWanted))/10 #number of workers strayed from self.workersWanted
        for i in range(0, 26):
            if self.inputs[i] < -1 or self.inputs[i] > 1:
                print(str(i) + ": " + str(self.inputs[i]))



    def propigate(self):
        outputs = []
        input_sums = []
        for i in range(0,self.hidden_nodes):
            node = self.weights[i]
            sum = 0
            for j in range(0, self.num_inputs):
                sum += self.inputs[j] * node[j]
            sum += self.weights[i][self.num_inputs]
            input_sums.append(sum)
            outputs.append(self.sigmoid(sum))
        sum = 0
        for i in range(0, self.hidden_nodes):
            sum += outputs[i] * self.weights[-1][i]
        sum += self.weights[-1][self.hidden_nodes]
        outputs.append(self.sigmoid(sum))
        input_sums.append(sum)
        return (outputs, input_sums)


    def sigmoid(self, x):
        return (1 / (1 + math.e ** (-1 * x)))

    def dsigmoid(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def backpropigation(self, eval):
        p = self.propigate()
        all_outputs = p[0]
        all_sums = p[1]

        # calculate the error and error term for the output
        output_error = eval - all_outputs[-1]
        self.ticks += 1
        if self.ticks % 10000 == 0:
            print("output error: " + str(output_error))
        output_error_term = output_error * self.dsigmoid(all_sums[-1])

        error_terms = []

        # calculate the error and error term for each hidden Node
        for i in range(0, self.hidden_nodes):
            node_error = output_error_term * self.weights[-1][i]
            node_error_term = node_error * self.sigmoid(all_sums[i])
            self.weights[-1][i] += self.learning_rate*node_error_term
            error_terms.append(node_error_term)

        # adjust the bias on the output node
        node_error = output_error_term * self.weights[-1][self.hidden_nodes]
        node_error_term = node_error * self.dsigmoid(all_sums[-1])
        self.weights[-1][self.hidden_nodes] += self.learning_rate*node_error_term

        for i in range(0, self.hidden_nodes):
            for j in range(0, self.num_inputs):
                node_error = error_terms[i] * self.weights[i][j]
                node_error_term = node_error * self.dsigmoid(all_sums[i])
                self.weights[i][j] += self.learning_rate*node_error_term*self.inputs[j]
            # adjust the bias
            node_error = error_terms[i] * self.weights[i][self.num_inputs]
            node_error_term = node_error * self.dsigmoid(all_sums[i])
            self.weights[i][self.num_inputs] += self.learning_rate*node_error_term






    def grow(self,temp,depth2):
        #print("here")
        if(depth2==self.depth):
            return 0
        if(temp.parent!=None): #if this was a bad move, don't epand it
            if(temp.util<temp.parent.util):
                return 0
        moves=listAllLegalMoves(temp.state)
        for move in moves: #check all possible moves utility
            newState=getNextState(temp.state,move)
            tempForTraining=self.getUtil(newState)
            self.getNetUtil(newState,tempForTraining)
            newNode=Node(newState,temp,move,tempForTraining)
            temp.addChild(newNode)
            self.grow(newNode,depth2+1)

    def findHighest(self,tempNode,highest):
        if(len(tempNode.children)>0): #if this isn't in the fringe, keep going
            for child in tempNode.children:
                self.findHighest(child,highest)
        else:
            if(tempNode.util>highest[1]): #see if this is better than our best
                highest[1]=tempNode.util
                last=tempNode
                cur=tempNode.parent
                while(cur.parent!=None):
                    last=cur
                    cur=cur.parent
                highest[0]=last.move

    def getUtil(self,cur):
        if(random.randint(0,10000)==0): #randomly can say this is a good move
            return 1.0
        myInv=getCurrPlayerInventory(cur)
        me=cur.whoseTurn
        #if(getWinner(cur)==me):
        #    return 1.0
        #elif(getWinner(cur)==1-me):
        #    return -1.0
        rating=140*(myInv.foodCount) # 140 points per food
        workers=getAntList(cur, me, (WORKER,))
        enemyQueen=getAntList(cur,1-me,(QUEEN,))
        enemyWorkers=getAntList(cur,1-me,(WORKER,))
        myQueen=getAntList(cur,me,(QUEEN,))[0]
        rating-=5*myQueen.coords[1] #take away points for having queen forward
        #if(myQueen.coords[1]>3):
        #   return -1.0
        if(len(enemyQueen)==0):
            rating+=1
        #    return 1.0
        else:
            rating-=enemyQueen[0].health*1000 #take away points for enemy queen health
        eworkers=getAntList(cur, 1-me, (WORKER,))
        myArmy=getAntList(cur, me, (SOLDIER,R_SOLDIER))
        enemyArmy=getAntList(cur,1-me,(DRONE,SOLDIER,R_SOLDIER))
        rating-=len(eworkers)*200
        diff=0
        for w in myArmy: #add points for my army
            diff+=1
            if(w.type==SOLDIER):
                diff-=2
                #rating+=100
            else:
                rating+=30
            rating+=w.health*UNIT_STATS[w.type][2]+UNIT_STATS[w.type][0]+UNIT_STATS[w.type][3]+200*UNIT_STATS[w.type][4]
            if(w.coords==self.enemyAnthill):
                rating+=1000
            elif(w.type==SOLDIER):
                rating+=20/approxDist(w.coords,self.enemyAnthill)
            elif(len(enemyWorkers)==0):
                dist=abs(2-approxDist(w.coords,self.enemyAnthill))+1
                rating+=20/dist
            elif(w.coords==self.enemyTunnel):
                rating+=30
            else:
                rating+=20/approxDist(w.coords,self.enemyTunnel)
        rating-=abs(diff)*100
        for w in enemyArmy: #take away points for the enemy army
            rating-=w.health*UNIT_STATS[w.type][2]+UNIT_STATS[w.type][0]+UNIT_STATS[w.type][3]+200*UNIT_STATS[w.type][4]
        for w in workers: #add points for workers
            if(w.carrying):
                rating+=55
                if(w.coords==self.myAnthill or w.coords==self.myTunnel):
                    rating+=110
                else:
                    close=approxDist(w.coords,self.myAnthill)
                    close2=approxDist(w.coords,self.myTunnel)
                    if(close2<close):
                        close=close2
                    rating+=40/close
            else:
                if(w.coords==self.myFood[0] or w.coords==self.myFood[1]):
                    rating+=55
                else:
                    close=approxDist(w.coords,self.myFood[0])
                    close2=approxDist(w.coords,self.myFood[1])
                    if(close2<close):
                        close=close2
                    rating+=20/close
        if(getAntAt(cur,self.myAnthill)!=None): #take away points if things are blocked
            rating-=30
        if(getAntAt(cur,self.myTunnel)!=None):
            rating-=30
        if(getAntAt(cur,self.myFood[0])!=None):
            rating-=30
        if(getAntAt(cur,self.myFood[1])!=None):
            rating-=30
        rating+=(self.workersWanted-abs(len(workers)-self.workersWanted))*1500 #adds points for the right number of workers
        #print(rating)
        if(rating>=999000):
            return .999
        if(rating<=-999000):
            return -.999
        #print(str(rating))
        return rating/1000000 #make this point system between -1 and 1
    ##
    #getAttack
    #
    # This agent never attacks
    #
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        return enemyLocations[0]  #don't care

    ##
    #registerWin
    #
    # This agent doens't learn
    #
    def registerWin(self, hasWon):
        self.printWeights()

    def printWeights(self):
        print("Printing Weights:")
        for weightL in self.weights:
            toPrint=""
            for indWeight in weightL:
                toPrint=toPrint+","+self.get3DecString(indWeight)
            print(toPrint)
        print("")

    def get3DecString(self, init):
        toRet=""
        temp=1000*init
        toRet=str(int(init))+"."
        return toRet+str(int(temp%1000))
