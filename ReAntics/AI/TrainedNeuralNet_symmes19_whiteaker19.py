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
from decimal import Decimal

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
        super(AIPlayer, self).__init__(inputPlayerId, "TrainedNeuralNet1")
        #the coordinates of the agent's food and tunnel will be stored in these
        #variables (see getMove() below)
        self.myFood = None
        self.myTunnel = None
        self.myAnthill=None
        self.workersWanted=3
        self.enemyAnthill=None
        self.enemyTunnel=None
        self.depth=2
        #self.learning_rate = .005
        self.num_inputs = 26
        self.hidden_nodes = int((self.num_inputs) * (2/3))
        self.ticks = 0
        self.inputs=[]
        for i in range(0,self.num_inputs):
            self.inputs.append(0)
        self.weights=[]
        self.setWeights();

    def setWeights(self):
        self.weights=[[-0.1063803,-0.0670141,0.9001399,-0.2746128,-0.305526,0.875269,0.4290594,0.5785365,0.8049606,-0.1143264,-0.8157229,0.4284387,-0.3015234,0.2213773,-0.6436223,-0.3167476,-0.129435,-0.4825639,-0.9247673,0.6901675,0.8440545,0.8740249,0.1932733,-0.1405605,-0.3457055,0.2153641,-1.081604],
                      [1.2322862,0.2287423,-0.0221313,0.475879,-0.1216321,0.0391268,-0.8075133,0.5633936,-0.6104749,0.6329353,-0.2381595,0.8687035,-0.3620895,-0.3225322,-0.1115065,-0.7100143,-0.2261322,-0.9826223,-0.0978538,0.1599672,-0.4508106,0.8575758,0.0554063,-0.3055887,0.8847331,-0.7128257,0.9506437],
                      [0.6887485,0.291809,-0.6354073,0.488001,-1.2896177,-0.7399136,0.0179731,-0.324172,-0.4063487,0.3844414,-0.1341894,0.0272087,0.5569871,-0.1625782,-0.7989598,-0.7929052,1.2421141,-0.6098863,0.3857504,-0.8003146,0.2275286,0.5271987,-0.3878716,-0.6411641,-0.8127867,0.6310025,0.4685757],
                      [-0.8883563,0.0362187,-0.7605942,0.8782218,-1.0128031,-0.8344622,0.3340044,0.1681696,-0.4486971,-0.3323945,0.1613262,0.4813388,0.5747528,-0.2278203,-0.2154153,0.8657575,0.7356784,0.0231692,-0.9104547,0.2940821,0.302122,0.4289438,0.8092136,0.6177419,0.9073448,0.2467696,-0.3096705],
                      [0.0939012,-0.0531743,0.1393293,0.0,0.0027449,-0.0354081,-0.7376575,-0.3829255,0.4831086,0.2222983,-0.288053,0.0004056,-0.2497193,0.3186965,0.1368381,0.345049,-0.138977,-0.0032044,0.0307729,-0.2569913,0.1518218,0.2596348,-0.8658448,0.6177336,0.8121722,-0.6587955,0.0000005],
                      [-0.1697374,-0.2762576,-0.3386961,0.0003653,0.142488,-0.9462135,0.6201845,-0.5316067,0.7505656,-0.3912946,-0.1212674,-0.2102067,0.241114,0.2812248,0.0356322,-0.0531195,-1.962095,0.0325956,-0.5386721,-0.5440339,0.163247,-0.5413736,-0.6839619,0.2514384,-0.3069845,-0.219602,0.0030246],
                      [1.5160723,1.2596623,-1.8271309,0.6164207,0.1514289,0.6211856,-0.0367245,-0.8162789,-0.6829045,-0.8310909,0.5654372,1.5699314,0.3591053,-0.5109237,-0.1787203,-0.002567,-0.0602927,1.2665457,-0.8201213,-0.2040893,-0.1414862,0.4898972,-0.8278762,0.404179,-0.6915389,0.949372,-0.2682019],
                      [-0.4251206,-0.7773189,0.3811128,0.0617343,-0.086009,-0.1882302,-0.1436743,-0.7539474,0.6443679,-0.0388901,0.3046885,-0.5396783,0.3626964,-0.6702874,0.7214688,-0.9015665,-0.7714492,0.3531499,-0.3311093,-0.1162638,0.8004211,0.0812957,0.7861341,0.0047598,-0.912935,0.7776452,-0.246584],
                      [0.4140542,0.8296327,-0.1006455,-0.312916,1.054739,-0.7229731,0.7660243,0.7439846,0.2180618,0.8804082,0.0737418,0.077948,-0.290846,-0.6482191,-0.0117733,-0.903544,-0.8359085,-0.4547241,0.7306298,-0.1486161,0.2406918,0.163978,-0.6035182,0.8063266,0.5181724,0.794899,-0.1488616],
                      [-0.0151499,-0.3127466,-0.8068579,0.3269575,-0.3758713,0.1994258,-0.4736709,-0.0295193,0.1230085,0.4008163,0.6060281,-0.2730629,0.9305622,0.2981464,0.7896688,0.1074963,0.4745896,-0.6903807,0.3621988,0.8982928,-0.7630043,-0.4252784,0.0355679,0.505616,0.251393,0.3380023,-0.6450734],
                      [-0.9013765,0.1690726,-0.2335605,-0.7816619,-0.9330928,0.860637,-0.3679284,-0.5101908,0.5772646,0.7114712,-0.7433729,1.0837479,-0.9932726,-0.7691539,-0.8620104,0.0183943,0.4798845,-0.335365,-0.305706,0.9710607,-0.0532687,0.6614876,-0.5202294,-0.3816513,0.4335346,0.8267555,0.0277041],
                      [-0.6955216,0.4263504,-0.0492543,0.1215059,0.4512026,-0.3561935,-0.3733461,0.5282611,-0.7806331,0.4097475,0.4299116,-0.5379926,-0.4397843,0.746442,0.9632887,-0.5503149,1.6934586,0.0445326,0.8159924,0.2064401,0.5800574,-0.4669646,-0.0454383,-0.6377946,0.4119361,-0.8270454,0.0861943],
                      [0.1779651,-0.7987607,-0.1561935,0.2590731,0.1122798,-0.1462496,0.4542162,0.8068089,-0.6656139,0.5050074,-0.8000349,-0.8005236,-0.1282621,1.1147825,-0.7785728,0.8794532,-0.0958642,0.781957,0.7430388,-0.6132256,0.0975051,0.6464822,-0.4546897,-0.9200672,-0.857589,0.765915,-1.0980032],
                      [0.9361693,-0.1130396,1.0864603,2.1490598,0.2105027,-0.5202983,0.8327231,-0.8328588,0.0476974,-0.0323228,-0.6833816,0.8503061,-0.5415502,0.875255,-1.036229,-0.8070333,0.0983261,-0.3349806,1.0181319,-0.9746534,-0.3404521,-0.7200851,0.1130611,0.6879543,0.874664,0.3789053,-0.2189088],
                      [-0.3940495,0.2784338,0.4713791,-0.0513866,0.5333669,-0.8570355,0.3185332,-0.3366629,-0.1380427,-0.1969351,0.2624903,-0.5209773,1.9149484,0.8048341,0.5316019,-0.0302201,-1.776323,0.1002118,-0.7893312,0.8838092,0.5815776,-0.8666757,0.3013079,0.1307797,0.8239695,0.3193835,-0.2437977],
                      [-1.0021238,-0.8743481,-0.0487693,-0.7058918,0.868751,-0.6794768,0.4064724,-0.6671448,0.9392628,-0.0545001,0.412733,-0.027966,-0.5681317,0.9460515,0.7046283,-0.8616746,0.4902197,0.5197905,0.1463904,-0.4594879,0.8950137,-0.001948,0.7630663,0.2853263,-0.982266,-0.6671384,-0.8138534],
                      [0.8635334,0.4600434,-1.2079632,0.322065,-0.0549157,-0.6510499,-0.601738,-0.6236855,0.746545,0.7895299,0.4698815,-2.0299867,-0.2504723,0.7234207,0.8926245,-0.4526158,-0.0464293,1.0973132,0.7031138,0.6031424,0.6191032,-0.2364016,0.4240199,0.4077219,-0.3179504,0.4895257,0.0886578],
                      [2.3966119,6.9439773,-3.7009616,-0.5210913,-51.6657413,-20.8960781,7.5139293,-2.332086,-2.8358497,-1.0310237,1.0156728,-1.381484,2.9208134,11.7705869,-5.1535031,0.2463293,5.1731377,-0.4320877]]
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
        #startUtil=self.getUtil(currentState)
        startUtil=self.getNetUtil(currentState)
        #print(str(startUtil))
        root=Node(currentState,None,None,startUtil)
        ourNode=root
        self.grow(root,0) #makes tree of possible states
        highest=[Move(END,None,None),startUtil]
        self.findHighest(root,highest) #finds best end state
        print("Highest: "+str(highest[0]))
        #print(str(highest[1]))
        return highest[0]

    def getNetUtil(self,currentState):
        self.setInputs(currentState)
        return self.propigate()

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
        return self.sigmoid(sum)


    def sigmoid(self, x):
        return (1 / (1 + math.e ** (-1 * x)))

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
            newNode=Node(newState,temp,move,self.getNetUtil(newState))
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
        pass
