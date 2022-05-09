import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from src.models.CTMC import *
from src.tools import randomProbabilities
from random import randint

def modelCTMC_random(nb_states: int, alphabet: list, min_waiting_time : int, max_waiting_time: int, self_loop: bool = True) -> CTMC:
	#lambda between 0 and 1
	s = []
	for j in range(nb_states):
		s.append([])
		for i in range(nb_states):
			if self_loop or i != j:
				s[j] += [i] * len(alphabet)
	if self_loop:
		obs = alphabet*nb_states
	else:
		obs = alphabet*(nb_states-1)

	states = []
	for i in range(nb_states):
		av_waiting_time = randint(min_waiting_time,max_waiting_time)
		states.append(CTMC_state([[p/av_waiting_time for p in randomProbabilities(len(obs))],s[i],obs]))

	return CTMC(states,0,"CTMC_random_"+str(nb_states)+"_states")

def modelCTMC1():
	s0 = CTMC_state([[0.05,0.45,0.5],[0,1,1],['a','a','b']])
	s1 = CTMC_state([[0.005,0.005],[0,1],['a','b']])
	return CTMC([s0,s1],0,"CTMC1")

def modelCTMC2(suffix=''):
	s0 = CTMC_state([[0.3/5,0.5/5,0.2/5],[1,2,3], ['r'+suffix,'g'+suffix,'r'+suffix]])
	s1 = CTMC_state([[0.08,0.25,0.6,0.07],[0,2,2,3], ['r'+suffix,'r'+suffix,'g'+suffix,'b'+suffix]])
	s2 = CTMC_state([[0.5/4,0.2/4,0.3/4],[1,3,3], ['b'+suffix,'g'+suffix,'r'+suffix]])
	s3 = CTMC_state([[0.95/2,0.04/2,0.01/2],[0,0,2], ['r'+suffix,'g'+suffix,'r'+suffix]])
	return CTMC([s0,s1,s2,s3],0,"CTMC2")

def modelCTMC3(suffix=''):
	s0 = CTMC_state([[0.65/4,0.35/4],[1,3],['g'+suffix,'b'+suffix]])
	s1 = CTMC_state([[0.6/3,0.1/3,0.3/3],[0,3,3],['g'+suffix,'g'+suffix,'b'+suffix]])
	s2 = CTMC_state([[0.25/5,0.6/5,0.15/5],[0,0,1],['r'+suffix,'g'+suffix,'b'+suffix]])
	s3 = CTMC_state([[1.0/10],[2],['g'+suffix]])
	return CTMC([s0,s1,s2,s3],0,"CTMC3")
