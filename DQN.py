import pygame
import sys
import random
import time
import numpy as np
import math
import collections
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import torch.optim as optim

pygame.init() 
ROW,COL = 5,5
width, height = 400,400 #keep width/ROW and height/COL %2 =0 !!!!
clock=pygame.time.Clock()
screen= pygame.display.set_mode((width,height))
class board:
	def __init__(self): 
		self.board = [[],[],[]]
	def draw_squares(self,screenn):
		screen.fill((255,255,255))
		for row in range(0,ROW+1,1):
			for col in range(0,COL+1,1):
				pygame.draw.rect(screen,(0,0,0),(row*(width/ROW),col*(width/ROW),width/ROW,height/COL),1)


def randomcell():
	r=random.randint(1,2*ROW-1)
	while r%2 ==0 and r!=0:
		r=random.randint(1,2*ROW-1)
	c=random.randint(1,2*COL-1)
	while c%2 ==0 and c!=0:
		c=random.randint(1,2*COL-1)
	random1=0.5*r*(width/ROW)
	random2=0.5*c*(height/COL)
	return random1,random2
#ADD CLASS HERE
def cellneighbour(s1,s2):
	#1 for going right, 2 left, 3 up and 4 down
	step=0
	while step!=1:
		step=0
		a=random.randint(1,4)
		if a==1:
			if s1<=width-((3/2)*width)/ROW:
				s1=s1+width/ROW
				step=step+1
				return s1,s2
			# else:
			# 	return s1,s2
		elif a ==2:
			if s1>=((3/2)*width)/ROW:
				s1=s1-width/ROW
				step=step+1
				return s1,s2
			# else:
			# 	return s1,s2
		elif a ==3:
			if s2>= ((3/2)*height)/COL:
				s2=s2-height/COL
				step=step+1
				return s1,s2
			# else:
			# 	return s1,s2
		elif a==4:
			if s2<=height-((3/2)*height)/COL:
				s2=s2+height/COL
				step=step+1
				return s1,s2
			# else:
			# 	return s1,s2

def updatescreen(coord):
	pygame.display.update()
	pygame.draw.circle(screen,(255,0,0),coord,10)
	pygame.display.update()
	pygame.draw.circle(screen,(255,255,255),coord,10)
	

def drawMaze():
	t=0
	L=[]
	LL=[]
	while len(L)<=ROW*COL-1:
		if t==0:
			n,k= randomcell()
			a,b=cellneighbour(n,k)
			updatescreen((n,k))
			c,d=n,k
			LL.append((n,k))
			LL.append((c,d))
			t=t+1
		elif t>=2:
			c,d=a,b
			a,b=cellneighbour(c,d)
			co1=(a,b)
			o=0
			while (co1 in L or (a==c and b==d)) and o<5:
				a,b=cellneighbour(c,d)
				co1=(a,b)
				o=o+1
			if o>=5:
				a,b=c,d 
		if a==c and b!=d:
			co1=(a,b)
			pygame.draw.line(screen,(255,255,255),(1.5+a-(width/(2*ROW)),((b+d)/2)),(-1.5+c+(width/(2*ROW)),((b+d)/2)),4)
			updatescreen((c,d))
			updatescreen((a,b))
			if len(L)<ROW*COL-1 or len(L)<ROW*COL-2 :
				pygame.draw.circle(screen,(255,255,255),(a,b),10)	
			LL.append((a,b))
			clock.tick(20)
			L.append(co1)
			t=t+1
		elif b==d and a!=c:
			co1=(a,b)
			pygame.draw.line(screen,(255,255,255),(((a+c)/2),1.5+b-(height/(2*COL))),(((a+c)/2),-1.5+b+(height/(2*COL))),4)
			updatescreen((c,d))
			updatescreen((a,b))
			if len(L)==ROW*COL-2:
				pygame.display.update()	
			clock.tick(10)
			L.append(co1)
			LL.append((a,b))
			t=t+1
		else:
			r=len(LL)-1
			if r>0:
				a,b=LL[r]
				LL.remove(LL[r])
	return screen
				


#memory buffer - create a class
#store stransition
#sample buffer
#Play episode
#input will be the position of the red circle

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #when we call Model() all the functions within init are initiated. 
        #2 numbers will discribe the position of the circle. 
        self.l1 = nn.Linear(2,30)
        self.l2 = nn.Linear(30, 60)
        self.l3= nn.Linear(60,4)
    #This is the action-value function, returns value for each action possible, we pick the highest.
    def forward(self, state):
    	#state can be one element or a batch.
    	state=state.to(T.device('cuda'))
    	x=F.relu(self.l1(state))
    	x=F.relu(self.l2(x))
    	x=self.l3(x)
    	return x
class DQN:
	capacity=100
	batch_size=100
	Gamma=0.999
	Target_update=5
	def __init__(self):
		#super(ClassName, self).__init__()
		self.end=[width-width/(2*ROW),height-height/(2*COL)]
		self.start=[width/(2*ROW),height/(2*COL)]
		self.D=collections.deque([],maxlen=self.capacity)
		self.target_mod, self.act_mod = Model().to(T.device('cuda')), Model().to(T.device('cuda'))
		#load parametrs initialized to target_mod()
		self.target_mod.load_state_dict(self.act_mod.state_dict())
		self.loss= nn.MSELoss()
		self.optimizer=optim.Adam(self.act_mod.parameters(), lr=0.0001)
	def store_trans(self,D,init_s,action,final_s,reward):
		init_s_=T.tensor(init_s)
		final_s_=T.tensor(final_s)
		D.append((init_s_,action,final_s_,reward))
	def States(self):
		S=[]
		for i in range(1,2*ROW,2):
			for j in range(1,2*COL,2):
				S.append((i*(width/(2*ROW)),j*(height/(2*COL))))
		return S
	def actionspace(self,coord):
		actionspace= {0:[width/ROW,0],1:[-width/ROW,0], 2:[0,-height/COL],3:[0,height/COL]}
		
		return actionspace
	def epsilonG(self,st,epsilon):
		st_=T.tensor(st)
		n=random.uniform(0,1)
		if n<epsilon:
			a=random.choice([0,1,2,3])
			return T.tensor(a).to(T.device('cuda'))
		else:
			#Calling the forward method with no_grad, we don't need this to optimize our model
			with T.no_grad():
				actions=self.act_mod(st_)
			a=T.argmax(actions).item()
			return T.tensor(a).to(T.device('cuda'))
	def movement(self,init_s,final_s):
		if init_s[0]==final_s[0]:
			#Same row different column
			return 'row'
		elif init_s[1]==final_s[1]:
			#Same column different row
			return 'column'
	def Ismovevalid(self,final_s):
		if final_s[0]<0 or final_s[1]<0 or final_s[0]>width or final_s[1]>height:
			return False
		else:
			return True
	def reward(self,init_s,final_s):
		#add reward for stepping on grey lines
		if final_s[0]==self.end[0] and final_s[1]==self.end[1]:
			reward= 100
			return T.tensor(reward).to(T.device('cuda'))
		elif self.movement(init_s,final_s)=='row':
			#sanctions for stepping on gray lines
			if screen.get_at((int(init_s[0]),int((init_s[1]+final_s[1])/2)))[:3]==(0,0,0):
				reward=-10
				return T.tensor(reward).to(T.device('cuda'))
			else:
				reward=-1
				return T.tensor(reward).to(T.device('cuda'))	
		elif self.movement(init_s,final_s)=='column':
			if screen.get_at((int((init_s[0]+final_s[0])/2),int(init_s[1])))[:3]==(0,0,0):
				reward=-10
				return T.tensor(reward).to(T.device('cuda'))
			else:
				reward=-1
				return T.tensor(reward).to(T.device('cuda'))
	def optimize_m(self):
		#not optimizing anything less than a batch size
		if len(self.D)<self.batch_size:
			return
		#function to sample batch 
		batch_=random.sample(self.D,self.batch_size)
		batch=list(zip(*[*batch_]))
		state_batch=T.stack(batch[0])
		action_batch=T.stack(batch[1]).unsqueeze(1)
		next_state_batch=T.stack(batch[2])
		reward_batch=T.stack(batch[3])
		#Model(state_batch) will give us equivalent values for the 4 actions,ie:
		#tensor [[value1,value2,value3,value4],[value1,value2,value3,value4]... number of elements in state batch]
		#.gather will give us the equivalent value of only the chosen action in action_batch [action1,action2,action3,action4 ... etc], so we can optimize
		state_action_values_Q = self.act_mod(state_batch).gather(1, action_batch)
		#Get the indexes of non final states
		next_state_indx=T.tensor([i for i in range(len(next_state_batch)) if (next_state_batch[i][0] != self.end[0] and next_state_batch[i][1] != self.end[1])]).to(T.device('cuda'))
		#indexes of final states
		zero_idx=T.tensor([i for i in range(len(next_state_batch)) if (next_state_batch[i][0] == self.end[0] and next_state_batch[i][1] == self.end[1])]).to(T.device('cuda'))
		target_values=self.target_mod(next_state_batch[next_state_indx]).max(1)[0]
		next_st_values=T.zeros(self.batch_size).to(T.device('cuda'))
		next_st_values[next_state_indx]=target_values
		if len(zero_idx)>0:
			next_st_values[zero_idx]=0
		Expected_Q = (next_st_values * self.Gamma) + reward_batch
		#.max(1)[0] would return a list like output containing max values, we still want it to look like state_action_values_Q
		# ie [[v1],[v2],[v3] ...]. So we use unsqueeze(1) to be able to get that shape.
		loss = self.loss(state_action_values_Q, Expected_Q.unsqueeze(1))
		#empty the grads for the new batch
		self.optimizer.zero_grad()
		#optimize by calc new weights and biases
		loss.backward()
		self.optimizer.step()
		return loss
	def playDQN(self):
		epsilon=1.0333333333 #Adjust as you please considering episode range
		#Initialize Q
		for episode in range(100):
			print(episode)
			#optimizer.zero_grad() We use zero_grad to allow for the new batch to have 0 grads
			init_s=self.start
			final_s=self.start
			ep=epsilon-0.0333333333
			while final_s!=self.end:
				action=self.epsilonG(init_s,ep)
				a=action.item()
				final_s=[S1+S2 for S1,S2 in zip(init_s,self.actionspace(init_s)[a])]
				if self.Ismovevalid(final_s)==False:
					final_s=init_s
					#keeps going back to while until it finds a valid move
					continue
				reward=self.reward(init_s,final_s)
				self.store_trans(self.D,init_s,action,final_s,reward)
				init_s=final_s
				self.optimize_m()
				if episode>60:
					updatescreen((final_s[0],final_s[1]))
					clock.tick(20)
			#update Target network every 5 episodes
			if episode%self.Target_update==0:
				self.target_mod.load_state_dict(self.act_mod.state_dict())
			if episode>self.batch_size and episode%20==0:
				print(self.optimize_m().item())
				
b=board()
b.draw_squares(screen)
drawMaze()
c=DQN()
c.playDQN()

running = True
while running:
  	for event in pygame.event.get():
	    if event.type == pygame.QUIT:
	    	running = False
	    if running == False:
	    	pygame.quit()



