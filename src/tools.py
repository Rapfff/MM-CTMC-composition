from random import random
from numpy.random import geometric
from ast import literal_eval

def loadSet(file_path:str) -> list:
	"""
	Load a training/test set saved into a text file.

	:param file_path: location of the file
	:type file_path: str

	:param float_obs: Should be True if the observations are float (False by default)
	:type float_obs: bool

	:return: a training/test set
	:rtype: float
	"""
	res_set = [[],[]]
	f = open(file_path,'r')
	l = f.readline()
	while l:
		res_set[0].append(literal_eval(l[:-1]))
		l = f.readline()
		res_set[1].append(int(l[:-1]))
		l = f.readline()
	f.close()
	return res_set

def saveSet(t_set: list, file_path: str) -> None:
	"""
	Save a training/test set into a text file.
	
	:param t_set: the training/test set to save
	:type t_set: list

	:param file_path: where to save
	:type file_path: str
	"""
	f = open(file_path,'w')
	for i in range(len(t_set[0])):
		f.write(str(t_set[0][i])+'\n')
		f.write(str(t_set[1][i])+'\n')
	f.close()

def resolveRandom(m: list) -> int:
	"""
	Given a list of probabilities it returns the index of the one choosen
	according to the probabilities.
	Example: if m=[0.7,0.3], it will returns 0 with probability 0.7, etc...
	
	:param m: list of probabilities
	:type m: list of float

	:return: an index
	:rtype: int
	"""
	while True:
		r = random()
		i = 0
		while r > sum(m[:i+1]) and i < len(m):
			i += 1
		if i < len(m):
			break
	return i

def randomProbabilities(size):
	"""return of list l of length <size> of probailities s.t. sum(l) = 1.0"""
	rand = []
	for i in range(size-1):
		rand.append(random())
	rand.sort()
	rand.insert(0,0.0)
	rand.append(1.0)
	return [rand[i]-rand[i-1] for i in range(1,len(rand))]

def generateSet(model,set_size,param,scheduler=None,distribution=None,min_size=None,timed=False):
	"""
	If distribution=='geo' then the sequence length will be distributed by a geometric law 
	such that the expected length is min_size+(1/param).
	if distribution==None param can be an int, in this case all the seq will have the same len (param),
					   or param can be a list of int
	"""
	seq = []
	val = []
	for i in range(set_size):
		if distribution == 'geo':
			curr_size = min_size + int(geometric(param))
		else:
			if type(param) == list:
				curr_size = param[i]
			elif type(param) == int:
				curr_size = param

		if scheduler:
			trace = model.run(curr_size,scheduler)
		elif timed:
			trace = model.run(curr_size,timed)
		else:
			trace = model.run(curr_size)


		if not trace in seq:
			seq.append(trace)
			val.append(0)

		val[seq.index(trace)] += 1

	return [seq,val]