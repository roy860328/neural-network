import tkinter as tk
import numpy as np

def setInterface():
	def clickTrain():
		# 
		lrate=learnrateentry.get()
		ccondition=convergenceentry.get()
		print ("learnrate:" + lrate)
		print ("convergence:" + ccondition)
		train(ccondition, lrate)

	interface = tk.Tk()
	#
	interface.title('interface')
	interface.geometry('500x500')
	#
	learnrate = tk.Label(interface, text="learnrate")
	learnrate.pack()
	learnrateentry = tk.Entry(interface)
	learnrateentry.pack()
	#
	convergence = tk.Label(interface, text="convergence")
	convergence.pack()
	convergenceentry = tk.Entry(interface)
	convergenceentry.pack()
	#
	trainbtn = tk.Button(interface, text="train", command=clickTrain)
	trainbtn.pack()
	#
	interface.mainloop()

def train(ccondition=100, lrate=0.5): 
	array = readFile()
	# print(array)
	
	inputx, weight1, weight2, outputy, row = setInitialization(array)

	#選擇2/3的隨機訓練data
	trainDatasIndex = np.random.choice(inputx.shape[0], size=int(row*2/3) + 1, replace=False)
	trainDatas = inputx[trainDatasIndex, :]
	print("trainDatas: ", trainDatas)
	#選擇1/3的隨機測試data
	testDatasIndex = np.arange(0, row)
	testDatasIndex = set(testDatasIndex) - set(trainDatasIndex)
	testDatasIndex = list(testDatasIndex)
	testDatas = inputx[testDatasIndex, :]
	print("testDatas: ", testDatas)
	#正確“訓練辨識”數量
	#正確“測試辨識”數量
	trainIdnumber = 0
	testIdnumber = 0

	############ start train ##############
	for n in range(ccondition):
		for i in range(trainDatas.shape[0]):
			####################one perceptron
			y1 = np.dot(weight1, trainDatas[i])
			###sgn[y1]
			if y1>0:
				y1 = 1
			else:
				y1 = 0
			###adjust weight1
			if y1 != outputy[trainDatasIndex[i]] and outputy[trainDatasIndex[i]] == 1:
				weight1 = weight1 + np.multiply(lrate, trainDatas[i])
				# print(weight1, n)
			elif y1 != outputy[trainDatasIndex[i]] and outputy[trainDatasIndex[i]] == 0:
				weight1 = weight1 - np.multiply(lrate, trainDatas[i])
				# print(weight1, n)
			####################
			####################two perceptron
			y2 = np.dot(weight2, trainDatas[i])
			###sgn[y2]
			if y2>0:
				y2 = 1
			else:
				y2 = 0
			###adjust weight2
			outy = outputy[trainDatasIndex[i]]
			if outy > 1:
				outy = 1
			else:
				outy = 0
			if y2 != outy and (outputy[trainDatasIndex[i]] == 2 or outputy[trainDatasIndex[i]] == 3):
				weight2 = weight2 + np.multiply(lrate, trainDatas[i])
				# print(weight2, n)
			elif y2 != outy and (outputy[trainDatasIndex[i]] == 0 or outputy[trainDatasIndex[i]] == 1):
				weight2 = weight2 - np.multiply(lrate, trainDatas[i])
				# print(weight2, n)
			####################
			###計算訓練辨識率
			y = y1 + y2*2
			# print(y, outputy[trainDatasIndex[i]], ' ', i)
			if y == outputy[trainDatasIndex[i]]:
				trainIdnumber = trainIdnumber + 1
	#訓練辨識率
	print("trainIdnumber: ", (trainIdnumber/trainDatas.shape[0])/ccondition)

	############ test rate ##############
	for i in range(testDatas.shape[0]):
		####################one perceptron
		y1 = np.dot(weight1, testDatas[i])
		###sgn[y1]
		if y1>0:
			y1 = 1
		else:
			y1 = 0
		####################
		####################two perceptron
		y2 = np.dot(weight2, testDatas[i])
		###sgn[y2]
		if y2>0:
			y2 = 1
		else:
			y2 = 0
		###adjust weight2
		outy = outputy[testDatasIndex[i]]
		if outy > 1:
			outy = 1
		else:
			outy = 0
		####################
		###計算測試辨識率
		y = y1 + y2*2
		print(y, outputy[testDatasIndex[i]], ' ', i)
		if y == outputy[testDatasIndex[i]]:
			testIdnumber = testIdnumber + 1
	#訓練辨識率
	print("testIdnumber: ", testIdnumber/testDatas.shape[0])
	print("weight1: ", weight1)
	print("weight2: ", weight2)

def setInitialization(array):
	row, col = array.shape
	###set up input weight1 bais
	#split inputx and outputy
	array = np.hsplit(array, [col-1])
	inputx = array[0]
	outputy = array[1]
	#add threshold to inputx
	threshold = np.zeros((row, 1)) -1
	inputx = np.hstack((threshold, inputx))
	#set weight1 bais
	weight1 = np.random.rand(1, col)
	weight2 = np.random.rand(1, col)

	return inputx, weight1, weight2, outputy, row

def network():
	x = 1

def readFile():
	try:
		pfile1 = open("perceptron2.txt", "r")
		string = pfile1.read()
		string = string.split('\n')
		#string to double list
		string = [i.split(' ') for i in string]
		string = [x for x in string if x != ['']]
		#string to float
		# try:
		# 	string = [[float(string2) for string2 in inner if string2] for inner in string]
		# except Exception as e:
		# 	raise e
		#string to 2D array
		string = np.array(string, dtype=float)
	except Exception as e: 
		print(e)

	return string

# setInterface()
train()