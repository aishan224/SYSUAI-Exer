import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from urllib import request
import time
import gzip
import os
import PIL
import cv2


filename = [
	["training_images","train-images-idx3-ubyte.gz"],
	["test_images","t10k-images-idx3-ubyte.gz"],
	["training_labels","train-labels-idx1-ubyte.gz"],
	["test_labels","t10k-labels-idx1-ubyte.gz"]
]

file_list = os.listdir()

def download_mnist():
	base_url = "http://yann.lecun.com/exdb/mnist/"
	for name in filename:
		print("Downloading "+name[1]+"...")
		request.urlretrieve(base_url+name[1], name[1])
	print("Download complete.")

def save_mnist():
	mnist = {}
	for name in filename[:2]:
		with gzip.open(name[1], 'rb') as f:
			mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
	for name in filename[-2:]:
		with gzip.open(name[1], 'rb') as f:
			mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
	with open("mnist.pkl", 'wb') as f:
		pickle.dump(mnist,f)
	print("Save complete.")

def init():
	download_mnist()
	save_mnist()

def load():
	with open("mnist.pkl",'rb') as f:
		mnist = pickle.load(f)
		# for i in range(10):
		# 	image = np.array(mnist["training_images"][1])
		# 	image = np.reshape(image,(28,28))
		# 	cv2.imshow('image%s'%i,image)
		# 	cv2.waitKey(0)
		# 	# plt.show()
		# 	# image = PIL.Image.fromarray(image)
		# 	# image.save("the%s.png" % i)

	return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def MakeOneHot(Y, D_out):
	N = Y.shape[0]
	Z = np.zeros((N, D_out))
	Z[np.arange(N), Y] = 1
	return Z

def draw_losses(losses):
	t = np.arange(len(losses))
	plt.plot(t, losses)
	plt.savefig('loss.png')
	# plt.show()

def get_batch(X, Y, batch_size):
	N = len(X)
	i = random.randint(1, N-batch_size)
	return X[i:i+batch_size], Y[i:i+batch_size]



class FC():   # FullyConnect
	"""
	Fully connected layer
	"""
	def __init__(self, D_in, D_out):
		#print("Build FC")
		self.cache = None
		#self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
		self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0} # weights
		self.b = {'val': np.random.randn(D_out), 'grad': 0}                                                      # bias 偏置

	def _forward(self, X):
		#print("FC: _forward")
		out = np.dot(X, self.W['val']) + self.b['val']           # WX+b
		self.cache = X
		return out

	def _backward(self, dout):
		#print("FC: _backward")
		X = self.cache
		dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
		self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
		self.b['grad'] = np.sum(dout, axis=0)
		#self._update_params()
		return dX

	def _update_params(self, lr=0.001):
		# Update the parameters
		self.W['val'] -= lr*self.W['grad']
		self.b['val'] -= lr*self.b['grad']

class ReLU():
	"""
	ReLU activation layer
	"""
	def __init__(self):
		#print("Build ReLU")
		self.cache = None

	def _forward(self, X):
		#print("ReLU: _forward")
		out = np.maximum(0, X)
		self.cache = X
		return out

	def _backward(self, dout):
		#print("ReLU: _backward")
		X = self.cache
		dX = np.array(dout, copy=True)
		dX[X <= 0] = 0
		return dX

class Sigmoid():
	"""
	Sigmoid activation layer # S型激活层
	"""
	def __init__(self):
		self.cache = None

	def _forward(self,X):
		self.cache = X
		return 1 / (1 + np.exp(-X))

	def _backward(self, dout):
		X = self.cache
		dX = dout*X*(1-X)
		return dX

class tanh():
	"""
	tanh activation layer
	"""
	def __init__(self,X):
		self.cache = X

	def _forward(self, X):
		self.cache = X
		return np.tanh(X)

	def _backward(self, X):
		X = self.cache
		dX = dout*(1 - np.tanh(X)**2)
		return dX

class Softmax():
	"""
	Softmax activation layer
	"""
	def __init__(self):
		#print("Build Softmax")
		self.cache = None

	def _forward(self, X):
		#print("Softmax: _forward")
		maxes = np.amax(X, axis=1)
		maxes = maxes.reshape(maxes.shape[0], 1)
		Y = np.exp(X - maxes)
		Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
		self.cache = (X, Y, Z)
		return Z # distribution

	def _backward(self, dout):
		X, Y, Z = self.cache
		dZ = np.zeros(X.shape)
		dY = np.zeros(X.shape)
		dX = np.zeros(X.shape)
		N = X.shape[0]
		for n in range(N):
			i = np.argmax(Z[n])
			dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
			M = np.zeros((N,N))
			M[:,i] = 1
			dY[n,:] = np.eye(N) - M
		dX = np.dot(dout,dZ)
		dX = np.dot(dX,dY)
		return dX

class Dropout():
	"""
	Dropout layer
	"""
	def __init__(self, p=1):
		self.cache = None
		self.p = p

	def _forward(self, X):
		M = (np.random.rand(*X.shape) < self.p) / self.p
		self.cache = X, M
		return X*M

	def _backward(self, dout):
		X, M = self.cache
		dX = dout*M/self.p
		return dX

class Conv():
	"""
	Conv layer
	"""
	def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):  # stride:步幅
		self.Cin = Cin
		self.Cout = Cout
		self.F = F
		self.S = stride
		#self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
		self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization  
		self.b = {'val': np.random.randn(Cout), 'grad': 0}
		self.cache = None
		self.pad = padding

	def _forward(self, X):
		X = np.pad(X,((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
		(N, Cin, H, W) = X.shape
		H_ = H - self.F + 1
		W_ = W - self.F + 1
		
		Y = np.zeros((N, self.Cout, H_, W_))

		for n in range(N):
			for c in range(self.Cout):
				for h in range(H_):
					for w in range(W_):
						Y[n, c, h, w] = np.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]
		self.cache = X
		return Y

	def _backward(self, dout):
		# dout (N,Cout,H_,W_)
		# W (Cout, Cin, F, F)
		X = self.cache
		(N, Cin, H, W) = X.shape
		H_ = H - self.F + 1
		W_ = W - self.F + 1
		W_rot = np.rot90(np.rot90(self.W['val']))

		dX = np.zeros(X.shape)
		dW = np.zeros(self.W['val'].shape)
		db = np.zeros(self.b['val'].shape)

		# dW
		for co in range(self.Cout):
			for ci in range(Cin):
				for h in range(self.F):
					for w in range(self.F):
						dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

		# db
		for co in range(self.Cout):
			db[co] = np.sum(dout[:,co,:,:])

		dout_pad = np.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
		#print("dout_pad.shape: " + str(dout_pad.shape))
		# dX
		for n in range(N):
			for ci in range(Cin):
				for h in range(H):
					for w in range(W):
						#print("self.F.shape: %s", self.F)
						#print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
						dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

		return dX

class MaxPool():   # 最大池化
	def __init__(self, F, stride):
		self.F = F
		self.S = stride
		self.cache = None

	def _forward(self, X):
		# X: (N, Cin, H, W): maxpool along 3rd, 4th dim
		(N,Cin,H,W) = X.shape
		# print(X.shape)
		F = self.F
		W_ = int(float(W)/F)
		H_ = int(float(H)/F)
		Y = np.zeros((N,Cin,H_,W_))
		M = np.zeros(X.shape) # mask
		for n in range(N):
			for cin in range(Cin):
				for h_ in range(H_):
					for w_ in range(W_):
						Y[n,cin,h_,w_] = np.max(X[n,cin,F*h_:F*(h_+1),F*w_:F*(w_+1)])
						i,j = np.unravel_index(X[n,cin,F*h_:F*(h_+1),F*w_:F*(w_+1)].argmax(), (F,F))
						M[n,cin,F*h_+i,F*w_+j] = 1
		self.cache = M
		return Y

	def _backward(self, dout):
		M = self.cache
		(N,Cin,H,W) = M.shape
		dout = np.array(dout)
		#print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
		dX = np.zeros(M.shape)
		for n in range(N):
			for c in range(Cin):
				#print("(n,c): (%s,%s)" % (n,c))
				dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
		return dX*M

def NLLLoss(Y_pred, Y_true):
	"""
	Negative log likelihood loss
	"""
	loss = 0.0
	N = Y_pred.shape[0]
	# print(Y_pred.shape,Y_true.shape)   # 修改处
	M = np.sum(Y_pred*Y_true, axis=1)
	for e in M:
		#print(e)
		if e == 0:
			loss += 500
		else:
			loss += -np.log(e)
	return loss/N

class CrossEntropyLoss():
	def __init__(self):
		pass

	def get(self, Y_pred, Y_true):
		N = Y_pred.shape[0]
		softmax = Softmax()
		prob = softmax._forward(Y_pred)
		loss = NLLLoss(prob, Y_true)
		Y_serial = np.argmax(Y_true, axis=1)
		dout = prob.copy()
		dout[np.arange(N), Y_serial] -= 1
		return loss, dout

class SoftmaxLoss():
	def __init__(self):
		pass

	def get(self, Y_pred, Y_true):
		N = Y_pred.shape[0]
		loss = NLLLoss(Y_pred, Y_true)
		Y_serial = np.argmax(Y_true, axis=1)
		dout = Y_pred.copy()
		dout[np.arange(N), Y_serial] -= 1
		return loss, dout

class Net(metaclass=ABCMeta):
	# Neural network super class

	@abstractmethod
	def __init__(self):
		pass

	@abstractmethod
	def forward(self, X):
		pass

	@abstractmethod
	def backward(self, dout):
		pass

	@abstractmethod
	def get_params(self):
		pass

	@abstractmethod
	def set_params(self, params):
		pass


class TwoLayerNet(Net):

	#Simple 2 layer NN

	def __init__(self, N, D_in, H, D_out, weights=''):
		self.FC1 = FC(D_in, H)
		self.ReLU1 = ReLU()
		self.FC2 = FC(H, D_out)

		if weights == '':
			pass
		else:
			with open(weights,'rb') as f:
				params = pickle.load(f)
				self.set_params(params)

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		return h2

	def backward(self, dout):
		dout = self.FC2._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

	def set_params(self, params):
		[self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params


class ThreeLayerNet(Net):

	#Simple 3 layer NN

	def __init__(self, N, D_in, H1, H2,H3,H4, D_out, weights=''):
		self.FC1 = FC(D_in, H1)
		self.ReLU1 = ReLU()
		self.FC2 = FC(H1, H2)
		self.ReLU2 = ReLU()
		self.FC3 = FC(H2, H3)

		self.ReLU3 = ReLU()
		self.FC4 = FC(H3,H4)

		self.ReLU4 = ReLU()
		self.FC5 = FC(H4, D_out)

		if weights == '':
			pass
		else:
			with open(weights,'rb') as f:
				params = pickle.load(f)
				self.set_params(params)

	def forward(self, X):
		h1 = self.FC1._forward(X)
		a1 = self.ReLU1._forward(h1)
		h2 = self.FC2._forward(a1)
		a2 = self.ReLU2._forward(h2)
		h3 = self.FC3._forward(a2)

		a3 = self.ReLU3._forward(h3)
		h4 = self.FC4._forward(a3)

		a4 = self.ReLU4._forward(h4)
		h5 = self.FC5._forward(a4)
		return h5

	def backward(self, dout):
		dout = self.FC5._backward(dout)
		dout = self.ReLU4._backward(dout)

		dout = self.FC4._backward(dout)
		dout = self.ReLU3._backward(dout)

		dout = self.FC3._backward(dout)
		dout = self.ReLU2._backward(dout)
		dout = self.FC2._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.FC1._backward(dout)

	def get_params(self):
		return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b, self.FC4.W, self.FC4.b, self.FC5.W, self.FC5.b]

	def set_params(self, params):
		[self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b, self.FC4.W, self.FC4.b, self.FC5.W, self.FC5.b] = params


class LeNet5(Net):
	# LeNet5

	def __init__(self,weights=''):
		# self.conv1 = Conv(1, 6, 5)
		self.conv1 = Convolution(1, 6, 5)
		self.ReLU1 = ReLU()
		# self.pool1 = MaxPool(2,2)
		self.pool1 = Pooling(2, 2)
		# self.conv2 = Conv(6, 16, 5)
		self.conv2 = Convolution(6, 16, 5)
		self.ReLU2 = ReLU()
		# self.pool2 = MaxPool(2,2)
		self.pool2 = Pooling(2,2)
		self.FC1 = FC(16*5*5, 120)
		self.ReLU3 = ReLU()
		self.FC2 = FC(120, 84)
		self.ReLU4 = ReLU()
		self.FC3 = FC(84, 10)
		self.Softmax = Softmax()

		self.p2_shape = None

		if weights == '':    #
			pass
		else:
			with open(weights,'rb') as f:   #只测试的时候带上参数，就能用之前训练的参数测试了
				params = pickle.load(f)
				self.set_params(params)
			

	def forward(self, X):
		h1 = self.conv1._forward(X)
		a1 = self.ReLU1._forward(h1)
		p1 = self.pool1._forward(a1)
		h2 = self.conv2._forward(p1)
		a2 = self.ReLU2._forward(h2)
		p2 = self.pool2._forward(a2)
		self.p2_shape = p2.shape
		# print(p2.shape)
		fl = p2.reshape(X.shape[0],-1) # Flatten  #new修改处
		h3 = self.FC1._forward(fl)
		a3 = self.ReLU3._forward(h3)
		h4 = self.FC2._forward(a3)
		a5 = self.ReLU4._forward(h4)
		h5 = self.FC3._forward(a5)
		# a5 = self.Softmax._forward(h5)  new修改处
		return h5

	def backward(self, dout):
		#dout = self.Softmax._backward(dout)
		dout = self.FC3._backward(dout)
		dout = self.ReLU4._backward(dout)
		dout = self.FC2._backward(dout)
		dout = self.ReLU3._backward(dout)
		dout = self.FC1._backward(dout)
		dout = dout.reshape(self.p2_shape) # reshape
		dout = self.pool2._backward(dout)
		dout = self.ReLU2._backward(dout)
		dout = self.conv2._backward(dout)
		dout = self.pool1._backward(dout)
		dout = self.ReLU1._backward(dout)
		dout = self.conv1._backward(dout)

	def get_params(self):
		return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

	def set_params(self, params):
		[self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params

class SGD():
	def __init__(self, params, lr=0.001, reg=0):
		self.parameters = params
		self.lr = lr
		self.reg = reg

	def step(self):
		for param in self.parameters:
			param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

def adam(w, dw, config=None):
	"""
	config format:
	- learning_rate: Scalar learning rate.
	- beta1: Decay rate for moving average of first moment of gradient.
	- beta2: Decay rate for moving average of second moment of gradient.
	- epsilon: Small scalar used for smoothing to avoid dividing by zero.
	- m: Moving average of gradient.
	- v: Moving average of squared gradient.
	- t: Iteration number.
	"""
	if config is None: 
		config = {}
	config.setdefault('learning_rate', 1e-3)
	config.setdefault('beta1', 0.9)
	config.setdefault('beta2', 0.999)
	config.setdefault('epsilon', 1e-8)
	config.setdefault('m', np.zeros_like(w))
	config.setdefault('v', np.zeros_like(w))
	config.setdefault('t', 0)
	
	m = config['m']
	v = config['v']
	t = config['t'] + 1
	beta1 = config['beta1']
	beta2 = config['beta2']
	epsilon = config['epsilon']
	learning_rate = config['learning_rate']

	m = beta1 * m + (1 - beta1) * dw
	v = beta2 * v + (1 - beta2) * (dw ** 2)
	mb = m / (1 - beta1 ** t)
	vb = v / (1 - beta2 ** t)
	next_w = w - learning_rate * mb / (np.sqrt(vb) + epsilon)

	config['m'] = m
	config['v'] = v
	config['t'] = t

	return next_w, config

class SGDMomentum(): 
	def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
		self.l = len(params)
		self.parameters = params
		self.velocities = []
		for param in self.parameters:
			self.velocities.append(np.zeros(param['val'].shape))
		self.lr = lr
		self.rho = momentum
		self.reg = reg
		self.config_list = []
		for i in range(self.l):
			self.config_list.append({})

	def step(self):
		for i in range(self.l):
			# self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
			# self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])
			self.parameters[i]['val'], self.config_list[i] = adam(self.parameters[i]['val'], self.parameters[i]['grad'], config=self.config_list[i])


def im2col(input_data, filter_h, filter_w, stride = 1, pad=0):
	"""
	Parameters
	------------
	input_data : 由（数据量，通道，高，长）的4维数组构成的输入数据
	fliter_h : 卷积核的高
	filter_w : 卷积核的长
	stride : 步幅
	pad : 填充

	Returns
	--------
	col : 2维数组
	"""
	# 输入数据的形状
    	# N：批数目，C：通道数，H：输入数据高，W：输入数据长
	N, C, H, W = input_data.shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride +1
	# 填充H,W
	img = np.pad(input_data, [(0,0),(0,0),(pad,pad),(pad,pad)], 'constant')
	#(N, C, filter_h, filter_w, out_h, out_w)的0矩阵
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
	# 按(0, 4, 5, 1, 2, 3)顺序，交换col的列，然后改变形状
	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
	return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
	N, C, H, W = input_shape
	out_h = (H + 2*pad - filter_h)//stride + 1
	out_w = (W + 2*pad - filter_w)//stride + 1
	col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

	img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
	for y in range(filter_h):
		y_max = y + stride*out_h
		for x in range(filter_w):
			x_max = x + stride*out_w
			img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

	return img[:, :, pad:H + pad, pad:W + pad]

class Convolution():
	# 初始化权重（卷积核4维）、偏置、步幅、填充
	def __init__(self, Cin, Cout, F, stride=1, pad=0, bias=True):
		# self.W = W
		# self.b = b
		self.Cin = Cin
		self.Cout = Cout
		self.F = F
		self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization  
		self.b = {'val': np.random.randn(Cout), 'grad': 0}
		self.stride = stride
		self.pad = pad
	
		# 中间数据（backward时使用）
		self.x = None   
		self.col = None
		self.col_W = None
	
		# 权重和偏置参数的梯度
		self.dW = None
		self.db = None

	def _forward(self, x):
		# 卷积核大小
		FN, C, FH, FW = self.W['val'].shape
		# 数据数据大小
		N, C, H, W = x.shape
		# 计算输出数据大小
		out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
		out_w = 1 + int((W + 2*self.pad - FW) / self.stride)
		# 利用im2col转换为行
		col = im2col(x, FH, FW, self.stride, self.pad)
		# 卷积核转换为列，展开为2维数组
		col_W = self.W['val'].reshape(FN, -1).T
		# 计算正向传播
		out = np.dot(col, col_W) + self.b['val']
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

		self.x = x
		self.col = col
		self.col_W = col_W

		return out

	def _backward(self, dout):
		# 卷积核大小
		FN, C, FH, FW = self.W['val'].shape
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)

		self.db = np.sum(dout, axis=0)
		self.dW = np.dot(self.col.T, dout)
		self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
		self.W['grad'] = self.dW
		self.b['grad'] = self.db

		W_rot = np.rot90(np.rot90(self.W['val'])).reshape(FN, -1).T

		dcol = np.dot(dout, W_rot.T)
		# 逆转换
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

		return dx

class Pooling:
	def __init__(self, pool_h, pool_w,stride=2, pad = 0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad

		self.x = None
		self.arg_max = None 
	
	def _forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)
		# 展开
		col = im2col(x,self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1,self.pool_h*self.pool_w)
		#最大值
		arg_max = np.argmax(col, axis=1)
		out = np.max(col,axis=1)
		#转换
		out = out.reshape(N, out_h, out_w, C).transpose(0,3,1,2)

		self.x = x
		self.arg_max = arg_max

		return out

	def _backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size, ))

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
		
		return dx 



"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""
start = time.time()

if('mnist.pkl' not in file_list):
	init()

X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train/float(255), X_test/float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

batch_size = 64
D_in = 784
D_out = 10

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
# H=400
# model = TwoLayerNet(batch_size, D_in, H, D_out)
H1 = 700
H2 = 500
H3 = 300
H4 = 100


def Train():
	# model = ThreeLayerNet(batch_size, D_in, H1, H2,H3,H4,D_out)
	model = LeNet5('weights.pkl')   #new 注释，训练的时候带上weights.pkl就是在原有weights的基础上再训练，不带就是从头来
					     # 测试的时候带上就可以用这个参数来测试测试集
	
	losses = []
	#optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
	optim = SGDMomentum(model.get_params(), lr=0.00008, momentum=0.80, reg=0.00003)  #修改处
	criterion = CrossEntropyLoss()


	# TRAIN
	ITER = 10000
	for i in range(ITER):
		# get batch, make onehot
		X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
		Y_batch = MakeOneHot(Y_batch, D_out)

		X_batch = np.reshape(X_batch,(batch_size,1,28,28))
		X_batch = np.pad(X_batch,((0,0),(0,0),(2,2),(2,2)),'constant')  # 修改处
		# forward, loss, backward, step
		Y_pred = model.forward(X_batch)
		loss, dout = criterion.get(Y_pred, Y_batch)
		model.backward(dout)
		optim.step()

		if i % 100 == 0:
			print("%s%% iter: %s, loss: %s" % (100*i/ITER,i, loss))
			nowtime = time.time()
			print("目前用时"+ str((nowtime-start)) + "s")
			losses.append(loss)


	## save params
	weights = model.get_params()
	with open("weights.pkl","wb") as f:
		pickle.dump(weights, f)

	draw_losses(losses)



	# # TRAIN SET ACC
	# X_train = np.reshape(X_train,(X_train.shape[0],1,28,28))
	# X_train = np.pad(X_train,((0,0),(0,0),(2,2),(2,2)),'constant')
	# Y_pred = model.forward(X_train)
	# result = np.argmax(Y_pred, axis=1) - Y_train
	# result = list(result)
	# print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(result.count(0)/X_train.shape[0]))

def Test():
	model = LeNet5('weights.pkl')
	global X_test
	global Y_test
	# TEST SET ACC
	# X_test = X_test[:5000]
	# Y_test = Y_test[:5000]
	X_test = np.reshape(X_test,(X_test.shape[0],1,28,28))
	X_test = np.pad(X_test,((0,0),(0,0),(2,2),(2,2)),'constant')
	Y_pred = model.forward(X_test)
	result = np.argmax(Y_pred, axis=1) - Y_test
	result = list(result)
	print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(result.count(0)/X_test.shape[0]))


# Train()
Test()
end = time.time()
print(end-start)
