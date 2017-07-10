import numpy as np
from util import get_robert_frost_data, init_weights
import theano
import theano.tensor as T

class SimpleRNN(object):
	def __init__(self, V, D, M):
		self.M = M
		self.D = D
		self.V = V

	def fit(self, X, lr=10e-4, mu=0.99):
		N = len(X)
		M = self.M
		D = self.D
		V = self.V


		#Initialize weights
		We = init_weights(V, D)
		Wx = init_weights(D, M)
		Wh = init_weights(M, M)
		bh = np.zeros(M).astype(np.float32)
		h0 = np.zeros(M).astype(np.float32)
		Wo = init_weights(M,V)
		bo = np.zeros(V).astype(np.float32)

		#Create all the theano variables and equations for training and prediction
		self.set(We, Wx, Wh, bh, h0, Wo, bo, np.float32(lr), np.float32(mu))


		#Stochastic Gradient Descent
		for n in range(2000):
			n_total=0
			n_correct=0
			tot_cost=0
			if n%10 == 0:
				lr *= 0.99
			for i in range(N):
				line = X[i]
				n_total+=len(line)
				in_seq = [0] + line
				out_seq = line + [1]
				#print(in_seq, out_seq)
				p, c = self.train(in_seq, out_seq)
				for i in range(len(p)):
					if p[i] == out_seq[i]:
						n_correct+=1
				tot_cost+=c
			print("iteration:", n, "Cost: ", tot_cost, "classification-rate:", float(n_correct)/n_total)
		self.save()


	def set(self, We, Wx, Wh, bh, h0, Wo, bo, lr=0, mu=0):
		M = self.M
		D = self.D
		V = self.V

		#Theano variables for weights and momentum parameters
		self.We = theano.shared(We)
		self.Wx = theano.shared(Wx)
		self.Wh = theano.shared(Wh)
		self.bh = theano.shared(bh)
		self.h0 = theano.shared(h0)
		self.Wo = theano.shared(Wo)
		self.bo = theano.shared(bo)

		self.params=(self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo)
		#dparams = [theano.shared(np.zeros(p.get_value().shape).astype(np.float32)) for p in self.params]

		thX = T.ivector('X')	
		thY = T.ivector('Y')		#vector of size T
		Ev = self.We[thX]			#T x D matrix (Embedded vector) 

		def recurrence(x_t, h_t_1):
			h_t = T.nnet.relu(x_t.dot(self.Wx) + h_t_1.dot(self.Wh) + self.bh)
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
		cost = -T.mean(T.log(pY[T.arange(thY.shape[0]), thY]))
		#cost = -(thY * T.log(pY).sum()

		grads = [ T.grad(cost, p) for p in self.params ]
		updates = [
				(p, p - lr*g) for p,g in zip(self.params, grads)
			] 
#+ [
#				(d, mu*d - lr*g) for d,g in zip(dparams, grads)
#			]'''

		#Training function - called by train_poetry
		self.train = theano.function(
			inputs=[thX, thY],
			outputs=[pred,cost],
			updates=updates
			)

		#Prediction function - called by generate_poetry
		self.predict = theano.function(
			inputs=[thX],
			outputs=pred,
			allow_input_downcast=True,
			)
	
	def save(self):
		print("save", self.bh)
		np.savez('rnn.npz', *[p.get_value() for p in self.params])

	@staticmethod
	def load():
		npzfile = np.load('rnn.npz')
		We = npzfile['arr_0']
		Wx = npzfile['arr_1']
		Wh = npzfile['arr_2']
		bh = npzfile['arr_3']
		h0 = npzfile['arr_4']
		Wo = npzfile['arr_5']
		bo = npzfile['arr_6']
	
		V, D = We.shape
		M, K = Wo.shape
		rnn = SimpleRNN(V, D, M)	
		rnn.set(We, Wx, Wh, bh, h0, Wo, bo)
		return rnn

	def generate_poetry(self, pi, word2idx):
		idx2word = {x:y for y,x in word2idx.items()}
		lineCount=0		#Print 4 lines
		V = len(idx2word)
		x = np.random.choice(V, p=pi)
		line = [0, x]
		print(idx2word[x],end=" ")
		while lineCount<4:
			p = self.predict(line)
			#print(p[0])
			if p[-1]>1:
				print(idx2word[p[-1]], end=" ")
				line += p[-1]
			else:
				#if(p[0] == 0):
				#print(p[0], "(start)")
				x = np.random.choice(V,p=pi)
				line = [x]
				print(p[-1])
				print(idx2word[x], end=" ")
				lineCount+=1



def train_with_data():
	sentences, word2idx = get_robert_frost_data()
	V=len(word2idx)
	M=30
	D=30
	model = SimpleRNN(V, D, M)
	model.fit(sentences)

def generate_new_poetry():
	sentences, word2idx = get_robert_frost_data()
	
	V = len(word2idx)
	pi = np.zeros(V)
	for s in sentences:
		pi[s[0]]+=1
	pi/=pi.sum()
	model = SimpleRNN.load()
	model.generate_poetry(pi, word2idx)

def main():
	train_with_data()
	generate_new_poetry()

if __name__ == '__main__':
	main()
