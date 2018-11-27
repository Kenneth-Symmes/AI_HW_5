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
        self.weights=[[0.6721581,0.2336765,0.4103827,-0.0137331,-0.4151352,0.1236353,0.071363,-0.6241001,0.3432808,0.6637554,-0.8393012,0.8448145,0.3039612,-0.5346411,-0.6771676,-0.3226989,-0.3316892,1.2730664,-0.247425,1.0312631,1.0254572,-0.1571942,0.8416269,0.8103183,-0.3308174,-0.9075404,0.7247903,-0.8680064],[-0.0663146,0.3787392,-0.2040871,-0.4300896,0.0813321,-0.0671498,-0.7771197,-0.6166414,-0.9250273,-0.5041606,0.001003,-0.3406211,0.5596149,-0.4524382,-0.9313529,0.31654,-0.6490255,0.615685,0.2237529,0.6371134,0.539887,0.6050524,-0.2732085,0.1088059,0.0961152,-0.5808505,0.8224368,1.1550563],[-0.0363232,0.1527838,-0.4474201,0.4477704,0.4244464,0.3714846,0.729466,0.040552,0.0613384,0.4678603,0.4398635,-0.5250572,0.3065499,-0.057135,-0.5067253,0.0220143,-0.9841614,-0.1993846,-0.7235773,0.5840877,-0.0158995,0.0243343,0.2550457,-0.1790098,-0.9239786,0.7782478,0.3890741,0.9654152],[0.6195082,0.4248269,0.3273573,1.0506224,1.0338555,-0.1990174,0.8737023,-0.6533909,0.4906454,-0.5906344,0.3793562,-0.9094505,-0.052731,0.4292829,0.9914264,0.225984,-0.488454,0.5712919,0.3392631,-0.8036112,-0.8915252,-0.625252,-0.5287175,0.379396,-0.3744284,0.0599711,-0.1611136,0.4924017],[-1.0159542,0.2509575,0.9821209,0.0353672,-0.5297747,0.3653238,0.3343278,-0.2883699,0.6580307,0.4401278,0.7612975,-0.2095158,0.787464,-0.4362272,-0.2606352,0.6103019,0.1614425,-1.1629323,-0.4552898,-0.3643191,-0.5501553,0.0711618,0.7634617,0.2637394,-0.2847093,0.1924689,-0.9022515,0.505326],[-0.2621466,-0.7230544,-0.6995196,0.1643906,-1.2282576,-0.6744097,-0.470277,-0.0008506,-0.7779454,0.6124487,0.1689548,-0.4486716,0.6611993,0.1901218,0.3102935,0.6478968,-0.3373906,1.8017481,-0.0442967,-0.1261634,0.8620326,0.2064117,0.1319692,-0.6666435,-0.8870699,0.1345924,0.0256554,0.2360387],[0.3534168,-0.4163779,-0.1080109,-0.298018,-0.6216417,-0.2446823,0.1701834,0.8191571,0.6198405,0.9443827,0.810301,-0.0576614,0.8508907,-0.5702111,0.7435487,-0.8164612,0.8402534,-1.2019433,0.0173759,-0.7660757,-0.4123811,-0.3078433,-0.3820665,-0.5796028,0.3508001,-0.4023318,-0.12574,0.8106518],[0.045342,-0.4358496,0.2333092,0.2528976,0.5475982,0.2909831,0.348503,-0.1991942,0.5205914,0.9223277,1.0049449,-0.6666295,0.0053399,-0.2212288,-0.9664276,0.783386,0.5071361,-1.2522156,-0.7310186,0.4612165,-0.47627,-0.2244968,-0.207727,-0.5387345,-0.8208849,-0.3815247,0.2642894,-0.4555591],[0.801694,0.7135533,0.6885989,0.5574145,0.6008617,-0.9254739,0.5657136,0.7326634,-0.9413186,0.0654685,-0.6013402,0.0302076,0.7195762,0.4298383,-0.340822,-0.5469565,-0.362108,-1.0286146,0.8506483,0.6167247,-0.6170548,-0.0583891,0.962667,-0.9352411,0.7352295,-0.5111422,-0.467486,0.5561603],[0.4857737,-0.1155453,0.0462798,-0.1467909,-1.35441,0.302661,0.6426442,0.907475,-0.455376,0.3508201,-0.6700608,-0.5696554,-0.0661782,0.593491,0.5871,0.3774813,0.4982682,1.7403265,0.4193206,-0.1777061,0.0487667,0.9556641,-0.3147037,0.4979823,-0.3892051,0.8960232,-0.0633858,0.2064546],[0.8872136,0.5398998,1.0656142,1.0206588,-0.6929727,0.0283256,0.7761037,0.8262676,0.8363789,-0.4744308,0.2197854,0.8132965,0.8587085,-0.2864292,-0.5998509,-0.7531554,-0.8170159,0.5983793,0.2307654,-0.4642911,-0.7439215,-0.0865239,-0.1118209,-0.29857,-0.2655363,0.5403097,0.042635,0.7879745],[0.6613307,0.6091144,-0.3293789,0.4333324,0.1983445,-0.9615643,0.0388729,-0.6007296,-0.121063,-0.0502107,-0.0945383,-0.2919028,0.52046,-0.9844644,0.70958,0.1116533,0.6479603,0.0128943,0.624522,0.2533144,-0.1333787,0.2297311,0.447481,-0.9488605,-0.3611222,-0.7579773,0.6137664,0.8610537],[-0.6534681,-0.1529411,0.6343242,-0.3760351,-1.4758972,0.0881515,-0.5594545,-0.2183993,-0.0284314,0.3699106,-0.4755732,-0.0807492,0.9469944,0.0214369,0.5114018,0.8268389,-0.240685,0.9217285,0.7821568,-0.2774256,-0.2653187,0.188405,-0.7905953,-0.9006005,0.5738116,0.2826285,0.5442909,-0.2646289],[0.9095535,-0.361915,0.4310383,-0.8989486,1.0148008,-0.5311402,-0.2140073,-0.7074003,-0.7494592,-0.8879825,0.4801021,0.6368084,0.94627,0.3713607,0.3953803,-0.219566,-0.0612393,-1.0783923,0.1986776,-0.0827113,-0.3924446,-1.0200003,0.1161024,-0.9014064,0.715705,0.8150146,-0.2107188,-0.9053302],[0.706974,-0.9962363,-0.7672388,0.586532,0.4107369,0.3646691,0.7337566,0.8419521,-0.1311444,-0.3923597,-0.1909501,-0.3039901,0.692088,0.9555371,-0.1308746,0.9032252,-0.7148428,0.1714224,-0.2134192,-0.6076491,0.5105473,0.7691963,0.1933621,0.2554789,-0.3518255,0.873694,0.4876125,0.2884552],[0.3849651,-0.2685392,0.206356,-1.0011369,-0.6420492,-0.7555239,0.7594521,0.1019323,0.195856,-0.3020883,-0.7176519,-0.8657524,0.2230056,0.232308,-0.7770543,0.8261599,-1.0617612,-0.1126685,-0.2704475,0.0628833,0.5170859,-0.7938575,0.6192001,0.3197851,0.423287,-0.7703322,0.6811658,0.3717171],[-0.4236321,-0.6796578,-0.4485562,-0.5804201,0.332585,0.9485271,-0.3967343,0.8790923,-0.7557302,0.59063,1.0541649,0.3136786,0.8320168,0.220505,-0.5150028,0.4764814,-0.5226167,-2.4418874,0.6487052,0.0574902,-0.3376728,0.5143484,-0.3326055,0.7922449,0.2711697,0.5339756,-0.1077514,-0.2882173],[0.4168812,-0.4678591,0.4232898,-0.0320731,0.5020233,-0.9915417,0.7325019,-0.8781632,-0.9651349,0.2712243,0.5409826,0.1776836,-0.1318269,-0.5428141,-0.2057591,-0.9230232,0.268265,0.7040339,-0.3335606,0.3011842,0.2157743,-0.2205611,-0.4049395,0.37304,-0.9092527,-0.2644148,0.9122779,-0.5050129],[1.7767104,0.0908582,0.1538613,-0.9919303,-0.809056,1.8471779,-0.2582199,-1.4677659,-0.3876809,1.7421312,0.20937,0.1215997,1.4180313,-0.6095329,-1.1188382,-0.3332708,-2.1377222,0.3994199,-0.2028676]]
        
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
