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
	
	inputx, outputy, row, col = setInitialization(array)

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
	##神經元(perceptron)數量
	y = np.zeros(int(np.amax(outputy))+1)
	weight = np.zeros(shape=(y.shape[0], col))
	for i in range(y.shape[0]):
		weight[i] = np.random.rand(1, col)

	############ start train ##############
	for n in range(ccondition):
		for i in range(trainDatas.shape[0]):

			for j in range(y.shape[0]):
				y[j] = calNetwork(weight[j], trainDatas[i])
				###adjust weight[j]
				weight[j] = adjustWeight(y[j], weight[j], outputy[trainDatasIndex[i]], lrate, 
																		trainDatas[i], j)
			####################
			###計算訓練辨識率
			# print(y, outputy[trainDatasIndex[i]], ' ', i)
			if getyresult(y, outputy[trainDatasIndex[i]]):
				trainIdnumber = trainIdnumber + 1
	#訓練辨識率
	print("traincorrectrate: ", (trainIdnumber/trainDatas.shape[0])/ccondition)

	############ test rate ##############
	for i in range(testDatas.shape[0]):

		for j in range(y.shape[0]):
			#################### perceptron
			y[j] = calNetwork(weight[j], testDatas[i])

		####################
		###計算測試辨識率
		# print(y, outputy[testDatasIndex[i]], ' ', i)
		if getyresult(y, outputy[trainDatasIndex[i]]):
			testIdnumber = testIdnumber + 1
	#訓練辨識率
	print("testcorrectrate: ", testIdnumber/testDatas.shape[0])
	[print("weight[", i, "]: ", weight[i]) for i in range(y.shape[0])]

def setInitialization(array):
	row, col = array.shape
	###set up input weight[j] bais
	#split inputx and outputy
	array = np.hsplit(array, [col-1])
	inputx = array[0]
	outputy = array[1]
	#add threshold to inputx
	threshold = np.zeros((row, 1)) -1
	inputx = np.hstack((threshold, inputx))

	return inputx, outputy, row, col

#calculate network and adjust result to two value
def calNetwork(weight, data):
	y = np.dot(weight, data)
	###sgn[y]
	if y>0:
		y = 1
	else:
		y = 0
	return y

#y=計算的結果, weight=當前權重, outputy=正確輸出, lrate=學習率, trainDatas=當前inputx, expectoutput=期望訓練的值 
def adjustWeight(y, weight, outputy, lrate, trainDatas, expectoutput):
	if y == 0 and outputy == expectoutput:
		weight = weight + np.multiply(lrate, trainDatas)
		# print(weight[j], n)
	elif y == 1 and outputy != expectoutput:
		weight = weight - np.multiply(lrate, trainDatas)
		# print(weight[j], n)
	return weight

# after calNetwork, if the result is correct
def getyresult(y, yRealValue):
	if y[int(yRealValue)] == 1:
		nonzero = np.nonzero(y)[0]
		if nonzero.shape[0] == 1:
			return True
	return False

def readFile():
	try:
		pfile1 = open("8OX.txt", "r")
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