import numpy as np
from util import get_robert_frost_data, init_weights
import theano
import theano.tensor as T
import random

class SimpleRNN(object):
	def __init__(self, V, D, M):
		self.M = M
		self.D = D
		self.V = V

	def fit(self, X, lr=0.1, mu=0.99):
		N = len(X)
		M = self.M
		D = self.D
		V = self.V


		#Initialize weights
		We = init_weights(V, D)
		Wx = init_weights(D, M)
		Wh = init_weights(M, M)
		bh = np.zeros(M).astype(np.float32)
		Wxz = init_weights(D, M)
		Whz = init_weights(M, M)
		bz = np.zeros(M).astype(np.float32)
		h0 = np.zeros(M).astype(np.float32)
		Wo = init_weights(M,V)
		bo = np.zeros(V).astype(np.float32)

		#Create all the theano variables and equations for training and prediction
		self.set(We, Wx, Wh, bh, Wxz, Whz, bz, h0, Wo, bo, np.float32(lr), np.float32(mu))


		#Stochastic Gradient Descent
		for n in range(450):
			n_total=0
			n_correct=0
			tot_cost=0
			if  n%100 == 0:
				lr *= 0.01
			for i in range(N):
				line = X[i]
				if random.random()<0.6:
					in_seq = [0] + line
					out_seq = line + [1]
				else:
					in_seq = [0] +line[:-1]
					out_seq = line
				n_total+=len(out_seq)
				p, c = self.train(in_seq, out_seq)
				for i in range(len(p)):
					if p[i] == out_seq[i]:
						n_correct+=1
				tot_cost+=c
			print("iteration:", n, "Cost: ", tot_cost, "classification-rate:", float(n_correct)/n_total)
		self.save()


	def set(self, We, Wx, Wh, bh, Wxz, Whz, bz, h0, Wo, bo, lr=0, mu=0):
		M = self.M
		D = self.D
		V = self.V

		#Theano variables for weights and momentum parameters
		self.We = theano.shared(We)
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.Wxz = theano.shared(Wx)
		self.Whz = theano.shared(Wh)
		self.bz = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)

		self.params=(self.We, self.Wx, self.Wh, self.bh, self.Wxz, self.Whz, self.bz, self.h0, self.Wo, self.bo)
		dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]

		thX = T.ivector('X')	
		thY = T.ivector('Y')		#vector of size T
		Ev = self.We[thX]			#T x D matrix (Embedded vector) 

		def recurrence(x_t, h_t_1):
			h_hat = T.nnet.relu(x_t.dot(self.Wx) + h_t_1.dot(self.Wh) + self.bh + np.float32(10e-200))
			z_t = T.nnet.sigmoid(x_t.dot(self.Wxz) + h_t_1.dot(self.Whz) + self.bz)
			h_t = z_t * h_hat + (1 - z_t) * h_t_1
			y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
			return h_t, y_t

		[h,y], _ = theano.scan(
			fn=recurrence,
			sequences=Ev,
			n_steps=Ev.shape[0],
			outputs_info=[self.h0,None],
			)

		# y -> T x D x V
		pY = y[:, 0, :]		# T x V
		pred = T.argmax(pY, axis=1)
		#np.place(num_py[T.arange(thY.shape[0]), thY]  , num_py[T.arange(thY.shape[0]), thY]  == 0, 10e-280)
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]+(10e-280)))
		#cost = -(thY * T.log(pY).sum()

		grads = [ T.grad(cost, p) for p in self.params ]
		updates = [
				(p, p - lr*g) for p,g in zip(self.params, grads)
			] 
#+ [
#				(d, mu*d - lr*g) for d,g in zip(dparams, grads)
#			]

		#Training function - called by train_poetry
		self.train = theano.function(
			inputs=[thX, thY],
			outputs=[pred,cost],
			updates=updates
			)

		#Prediction function - called by generate_poetry
		self.predict = theano.function(
			inputs=[thX],
			outputs=[pred],
			allow_input_downcast=True,
			)
	
	def save(self):
		print("save", self.bh)
		np.savez('rrnn.npz', *[p.get_value() for p in self.params])

	@staticmethod
	def load():
		npzfile = np.load('rrnn.npz')
		We = npzfile['arr_0']
		Wx = npzfile['arr_1']
		Wh = npzfile['arr_2']
		bh = npzfile['arr_3']
		Wxz = npzfile['arr_4']
		Whz = npzfile['arr_5']
		bz = npzfile['arr_6']
		h0 = npzfile['arr_7']
		Wo = npzfile['arr_8']
		bo = npzfile['arr_9']
	
		V, D = We.shape
		M, K = Wo.shape
		rnn = SimpleRNN(V, D, M)
		rnn.set(We, Wx, Wh, bh, Wxz, Whz, bz, h0, Wo, bo)
		return rnn

	def generate_poetry(self, pi, idx2word):
		#call the predict function to generate sequences of sentences
		lineCount=0

		V = len(idx2word)
		print(V)
		x = np.random.choice(V, p=pi)
		line = [x]
		#print("load", self.bh.get_value())
		#print("initial: ", [x])
		#print(idx2word[0])
		print(idx2word[x],end=" ")
		while lineCount<4:
			print(line)
			p = self.predict(line)[-1]
			#print(p[-1], end="")
			if p[-1]>1:
				print(idx2word[p[-1]], end=" ")
				line += [p[-1]]
			else:
				#if(p[0] == 0):
				print(p[-1], "(start)")
				x = np.random.choice(V,p=pi)
				line = [x]
				print(idx2word[x], end=" ")
				lineCount+=1



def train_with_data():
	sentences, word2idx = get_robert_frost_data()
	V=len(word2idx)
	M=50
	D=50
	model = SimpleRNN(V, D, M)
	model.fit(sentences)

def generate_new_poetry():
	sentences, word2idx = get_robert_frost_data()
	idx2word = {x:y for y,x in word2idx.items()}
	
	V = len(word2idx)
	print(V)
	pi = np.zeros(V)
	for s in sentences:
		pi[s[0]]+=1
	pi/=pi.sum()
	model = SimpleRNN.load()
	model.generate_poetry(pi, idx2word)

def main():
	train_with_data()
	generate_new_poetry()

if __name__ == '__main__':
	main()
