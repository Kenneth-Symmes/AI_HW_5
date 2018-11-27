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
        self.num_inputs = 27
        self.hidden_nodes = int((self.num_inputs) * (2/3))
        self.ticks = 0
        self.inputs=[]
        for i in range(0,self.num_inputs):
            self.inputs.append(0)
        self.weights=[]
        self.setWeights();

    def setWeights(self):
        self.weights=[[0.8886318,0.5337342,0.536702,-0.7985816,-1.1905497,0.396604,-0.1118418,0.3339852,-0.8153889,0.5100199,0.9250492,0.1401039,0.4063328,0.1457505,0.9634727,0.4933027,-0.3350897,-0.1254317,0.4652506,-0.1760088,-0.5237846,0.0001327,-0.2080429,0.4816634,-0.7586594,-0.6281681,-1.0364955,-0.0299871],[-0.3311155,0.097785,0.2433866,-1.075107,-0.3021376,0.3242163,0.6649483,0.897265,0.2474271,0.5901066,0.281838,-0.3263985,-1.2160257,0.3352156,0.5287868,0.0119823,0.376267,0.0060318,0.6834223,-0.5294451,-0.4127016,-0.3867303,0.2063098,0.5957155,0.907808,-0.1010999,-0.3777413,-1.4017276],[0.3000308,-0.802123,0.1064541,-1.0239092,-0.876497,-1.0275689,0.1294342,-0.731684,0.3706868,0.698812,0.0292166,-0.6189005,-1.1057085,-0.2190133,-0.138428,-0.2903007,-0.0620757,1.15342,-0.1980925,0.5720378,-0.4859006,-0.6118707,0.3773848,-0.3979208,-0.0663435,-0.9061271,-0.6843887,-0.1391348],[-0.5281375,-0.003203,0.4688498,-0.0321693,-1.9866649,-0.7620673,-0.0608477,-0.7775681,0.9565475,-0.0416794,0.2897844,0.7611358,0.1807093,-0.6041483,0.6537581,-0.1634811,0.3861469,1.6609572,1.4113012,0.3698139,-0.2574064,0.4356612,-1.2084017,-1.2184273,-0.1879397,0.4883275,-1.1913749,0.1583745],[1.2085621,-0.8710059,-0.502302,0.5832881,-1.714902,-0.7316309,1.1003963,-0.5138235,0.5952095,1.1808661,0.4542503,0.6515545,-0.1230513,-0.0920536,0.8692197,-1.0774715,0.029534,1.0585871,0.9296259,0.1038571,1.2353669,0.5842737,-0.3426706,-0.7917797,-0.1189913,0.3595024,-0.5629616,-0.3279078],[-0.3601234,-0.7469052,0.3045991,-1.2786526,-1.4364672,-0.5191739,0.2424901,0.492616,0.409988,-0.355978,0.2963335,0.691601,-0.6069369,0.0626264,0.5197426,-0.6482247,-0.5107036,2.1198361,-0.2690545,-0.0490515,-0.276404,0.0162628,0.5145509,-0.8589126,-0.9861529,-0.9079951,0.0167925,-0.4557238],[0.8782432,-0.9971637,-0.8487476,-0.0658516,0.0478173,0.2707778,0.6345683,0.7495506,-0.9269207,0.715607,0.89043,0.165133,-1.103152,0.1620834,0.3775081,-0.0306869,-0.5179959,-1.4921048,-0.1339937,0.2219346,-0.6866018,0.1619414,-0.6021569,-0.5093445,-0.2140178,-0.4335272,-1.8246884,-0.1048284],[-0.114321,0.2788221,0.6903837,-0.6446338,0.0712841,-0.4840139,0.2681548,0.2936504,0.8587694,0.6585222,0.339971,0.342092,0.4851864,0.350889,-0.8046338,0.6650754,-0.4425501,-1.6022927,-0.5709469,0.5260177,-0.1766473,-0.6585051,-0.837691,0.4646812,1.0384504,-0.3342538,-0.1049223,-1.6377364],[0.4550277,-0.6495848,-0.3129612,-0.8559132,-1.3914513,-0.8285447,-0.3101224,-0.5662743,0.7450265,0.6421563,0.5018916,0.0923189,-0.1113376,0.6167253,-0.5031293,-0.7378126,-0.5307912,-0.8017645,0.3486045,0.8626812,-0.9304166,-1.0122329,0.2712423,0.9406245,-0.2009319,0.8129753,-0.9036098,0.3277041],[0.9293006,-0.9865598,0.2892423,-1.3256535,-0.5052076,-0.5567008,0.5310754,-0.5378897,-0.1588795,0.5788155,-0.4197811,-0.0589732,-0.9694391,0.4905669,-0.7181053,-0.2283662,-0.2180447,0.3023701,-0.4163567,-0.2272185,0.5968561,-0.9643913,0.3289047,-0.6740156,0.9367969,0.1124332,-0.8536903,-0.0486611],[-0.2902165,0.6156402,0.0034021,0.2342943,0.4324069,-0.226459,0.7152814,0.4508379,-0.1791342,0.5624166,0.783194,0.2467747,-0.5020533,0.6618188,0.3145403,0.6501316,0.0214779,-3.0890094,-1.0349244,-0.4402252,-1.2007752,-1.0383248,0.4938544,-0.2152315,-0.3322709,-0.5789327,-1.3079619,-0.981147],[-0.2249442,-0.8864287,-0.7393648,-0.1473798,0.6550963,0.3518851,0.4599431,0.1374449,-0.4086897,-0.4912025,-0.7650684,-0.9316064,0.5540122,-0.2292987,0.0459761,0.3844796,-0.3459555,-2.4259979,0.5918763,-0.3587202,0.0776213,-0.5159427,-0.5877599,-0.4639826,0.2564233,0.0529485,-1.4185039,-0.6226184],[0.740104,-0.3612828,0.2153038,-0.234938,-2.6068261,-0.4998342,-0.123617,-0.8365875,0.432731,0.1655931,-0.4419275,-0.36573,0.1940595,0.5689203,-0.3176503,-0.0801869,-0.2437157,3.1643012,0.096668,0.9286299,-0.5223896,1.0064449,-0.0523791,-0.2283028,0.2146133,-0.0256338,-0.1551879,-0.1150604],[-0.5230429,-0.205002,-0.1248736,-0.3454286,-1.4098436,-0.106861,-0.7656885,-0.2875242,0.6108918,-0.0510491,0.8363241,0.0461083,-0.0826693,0.5810895,0.9655796,-0.1550546,0.2039045,0.3119058,0.0864488,0.2053335,-0.3728339,-0.3222717,-0.4612072,-0.0525236,0.107946,-0.4265245,-0.5032888,-1.0851704],[-0.3768648,-0.1799961,0.0105911,-0.4723977,0.306378,0.084422,-0.6913078,0.521222,-1.1168904,-1.1144879,-0.6578116,-0.1336379,0.313708,-0.3064855,-0.6012077,-0.3624912,0.4185539,-4.0977349,-0.5509845,-0.5540637,-0.2399432,-0.9976586,0.9876939,0.5128991,0.2957886,0.3681897,-0.0343398,-0.6178547],[-0.2605118,0.0754094,0.0180788,-1.8025999,2.171249,0.6875029,0.1057487,-1.2218671,0.3132678,-1.0238153,-0.4690481,-0.5861416,-0.0478754,-0.1061542,-0.9423619,0.3503054,-0.1412711,-4.8113279,-1.1448782,0.2630766,-1.0154737,0.0664246,-0.0022805,1.1919214,-0.3259814,0.8791839,-0.4447877,-0.5815606],[-1.1065122,-0.1080577,0.4454801,-0.2721366,-1.3645314,0.0321226,0.1814185,0.425963,-0.2665607,-0.3883091,0.2027545,-0.441309,-0.9956356,-0.8395116,0.1394652,1.0470435,-0.2723114,-1.0599363,0.0402329,-0.3054363,-0.6929938,0.7852078,0.511555,0.7357892,0.4234444,0.9742468,-0.8953586,-0.2097308],[-1.4779393,0.4510596,-0.2650651,-0.3398872,-1.6756095,-0.4508707,1.1874336,0.2291421,0.1375967,1.1971086,-0.64571,-0.3318571,0.075632,-0.739465,0.5330553,0.6376082,-0.083424,2.0655515,1.3018157,-0.3979156,0.9838495,-0.5444349,-0.7798045,0.2605583,-0.2553887,0.5136577,-0.6430995,-0.1436575],[0.4516032,0.0081423,0.5267777,1.1988371,1.5459482,0.9227231,-0.4530571,-0.6898306,-0.2582728,0.1258537,-1.3037973,-1.1127432,1.0512618,0.3114815,-1.8056618,-2.5333691,-0.3364234,1.2647186,-0.5078298]]



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
