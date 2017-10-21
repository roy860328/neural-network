import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import os
import sys

def setInterface():
	def clickTrainBtn():
		###clear 之前的plot
		plt.clf()
		###偵測列表選取的txt檔案   #################################listTxt如何取得的？
		selectionfile = listTxt.curselection()
		selectionfile = listTxt.get(selectionfile)
		array = readFile(selectionfile)
		###取得學習率跟收斂條件
		lrate=float(learnrateentry.get())
		ccondition=int(convergenceentry.get())

		trainrate, testrate, weight = train(array, ccondition, lrate)
		showTrainresult(trainrate, testrate, weight)
		plt.show()

	def showTrainresult(trainrate, testrate, weight):
		string = "\n" + "trainrate: " + str(trainrate) + "\n" + "testrate: " + str(testrate) + "\n" 
		# [(string += "weight[" + str(i) + "]: " + str(weight[i]) + "\n") for i in range(weight.shape[0])]
		for i in range(weight.shape[0]):
			string += "weight[" + str(i) + "]: " + str(weight[i]) + "\n"
		outputresultprint.set(string)

	interface = tk.Tk()
	#創造視窗x
	interface.title('interface')
	interface.geometry('800x800')
	#學習率字幕
	learnrate = tk.Label(interface, text="learnrate")
	learnrate.pack()
	#input learnrate
	learnrateentry = tk.Entry(interface)
	learnrateentry.insert(0, "0.5")
	learnrateentry.pack()
	#收斂字幕
	convergence = tk.Label(interface, text="convergence (train times)")
	convergence.pack()
	#input convergence
	convergenceentry = tk.Entry(interface)
	convergenceentry.insert(0, "10")
	convergenceentry.pack()
	#列出txt檔案
	listTxt = tk.Listbox(interface)
	##############os.path.dirname(sys.executable)當產出exe檔時才能正確找到txt檔案位置,但無法在.py檔中使用
	##############os.getcwd()只有在.py檔有用,因為exe檔的默認位置在"cd ~" 讀檔時會找不到檔案
	print("sys.executable directory: ", os.path.dirname(sys.executable))
	os.chdir(os.path.dirname(sys.executable))
	haveTxt = ''
	for file in os.listdir(os.path.dirname(sys.executable)):
		if file.endswith(".txt") or file.endswith(".TXT"):
			haveTxt += str(file) + ','
	haveTxt = haveTxt.split(",")
	haveTxt = list(filter(None, haveTxt))
	for txt in haveTxt:
		listTxt.insert(0, txt)
	# listTxt.bind('<<ListboxSelect>>', fileSelection)
	listTxt.pack()
	#訓練按鈕
	trainbtn = tk.Button(interface, text="train", command=clickTrainBtn)
	trainbtn.pack()
	#outputresult
	outputresult = tk.Label(interface, text="outputresult")
	outputresult.pack()
	#input learnrate
	outputresultprint = tk.StringVar()
	outputresultLabel = tk.Label(interface, textvariable=outputresultprint)
	outputresultLabel.pack()
	#
	#讓視窗實現
	interface.mainloop()

def train(array, ccondition=100, lrate=0.5): 
	
	inputx, outputy, row, col = setInitialization(array)

	#選擇2/3的隨機訓練data
	trainDatasIndex = np.random.choice(inputx.shape[0], size=int(row*2/3) + 1, replace=False)
	trainDatas = inputx[trainDatasIndex, :]
	#選擇1/3的隨機測試data
	testDatasIndex = np.arange(0, row)
	testDatasIndex = set(testDatasIndex) - set(trainDatasIndex)
	testDatasIndex = list(testDatasIndex)
	testDatas = inputx[testDatasIndex, :]
	#正確“訓練辨識”數量
	#正確“測試辨識”數量
	trainIdnumber = 0
	testIdnumber = 0
	##神經元(perceptron)outputy數量 weight初始化
	y = np.zeros(int(np.amax(outputy))+1)
	weight = np.zeros(shape=(y.shape[0], col))
	for i in range(y.shape[0]):
		weight[i] = np.random.rand(1, col)
	##儲存最後一次outputy結果，用來畫出圖形
	trainOutputResult = np.zeros(trainDatas.shape[0])
	testOutputResult = np.zeros(testDatas.shape[0])

	############ start train ##############
	for n in range(ccondition):
		for i in range(trainDatas.shape[0]):

			for j in range(y.shape[0]):
				y[j] = calNetwork(weight[j], trainDatas[i])
				###adjust weight[j]
				weight[j] = adjustWeight(y[j], weight[j], outputy[trainDatasIndex[i]], lrate, 
																		trainDatas[i], j)
			
			###計算訓練辨識率
			if judgeYResult(y, outputy[trainDatasIndex[i]]):
				trainIdnumber = trainIdnumber + 1
				##紀錄最後一次outputy結果
				if n == ccondition-1:
					trainOutputResult[i] = outputy[trainDatasIndex[i]]
			##紀錄最後一次outputy結果
			elif n == ccondition-1:
				trainOutputResult[i] = -1

	#print訓練辨識率
	print("traincorrectrate: ", (trainIdnumber/trainDatas.shape[0])/ccondition)

	############ test rate ##############
	for i in range(testDatas.shape[0]):

		for j in range(y.shape[0]):
			y[j] = calNetwork(weight[j], testDatas[i])

		###計算測試辨識率
		if judgeYResult(y, outputy[testDatasIndex[i]]):
			testIdnumber = testIdnumber + 1
			##紀錄最後一次outputy結果
			if n == ccondition-1:
				testOutputResult[i] = outputy[testDatasIndex[i]]
		##紀錄最後一次outputy結果
		elif n == ccondition-1:
			testOutputResult[i] = -1

	#print測試辨識率 和 weight[]
	print("testcorrectrate: ", testIdnumber/testDatas.shape[0])
	[print("weight[", i, "]: ", weight[i]) for i in range(y.shape[0])]

	#如果沒用到weight[0]，則設成0
	if int(np.amin(outputy)) != 0:
		weight[0] = [0]
	##########畫出圖形##########
	try:
		plt.figure(1, figsize=(16, 12))
		plt.subplot(221)
		plt.title("TrainSample")
		showPlot(trainDatas, trainOutputResult, y.shape[0], weight)
		plt.subplot(222)
		plt.title("TestSample (black point is identify error data)")
		showPlot(testDatas, testOutputResult, y.shape[0], weight)
	except Exception as e:
		pass

	return ((trainIdnumber/trainDatas.shape[0])/ccondition), (testIdnumber/testDatas.shape[0]), weight

#Initialize the text file to inputx and outputy array
def setInitialization(array):
	row, col = array.shape
	###set up inputx and outputy
	#split inputx and outputy
	array = np.hsplit(array, [col-1])
	inputx = array[0]
	outputy = array[1]
	#add threshold to inputx
	threshold = np.zeros((row, 1)) -1
	inputx = np.hstack((threshold, inputx))

	return inputx, outputy, row, col

#calculate network and adjust result to two value (0 or 1)
def calNetwork(weight, datax):
	y = np.dot(weight, datax)
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
	elif y == 1 and outputy != expectoutput:
		weight = weight - np.multiply(lrate, trainDatas)
	return weight

# after calNetwork, if the result is correct  return True, else False
def judgeYResult(y, yRealValue):
	if y[int(yRealValue)] == 1:
		nonzero = np.nonzero(y)[0]
		if nonzero.shape[0] == 1:
			return True
	return False

#可以plot出2D的data. outputResult為資料集輸出的結果(output). 
def showPlot(Datas, outputResult, outputHadvalue, weight):
	Datas = np.hsplit(Datas, [1])
	Datas = np.hsplit(Datas[1], [1])

	# plt.scatter(Datas[0], Datas[1], c='r', label='perceptron1')
	pointlabel = np.zeros(outputHadvalue)
	colorSelect = ['black', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
	for i in range(Datas[0].shape[0]):
		plt.scatter(Datas[0][i], Datas[1][i], c=colorSelect[int(outputResult[i])+1], label=str(int(outputResult[i])) if pointlabel[int(outputResult[i])] == 0 else "")
		if pointlabel[int(outputResult[i])] == 0:
			pointlabel[int(outputResult[i])] = 1

	plt.xlim([-5, 5])
	plt.ylim([-5, 8])

	x = np.arange(-5, 5, 0.1)
	for i in range(weight.shape[0]):
		y = -(weight[i][1]/weight[i][2])*x - weight[i][0]/weight[i][2]
		plt.plot(x,y)
	plt.legend()
	
#read text file
def readFile(file):
	try:
		pfile1 = open(file, "r")
		string = pfile1.read()
		string = string.split('\n')
		#string to double list
		string = [i.split(' ') for i in string]
		string = [x for x in string if x != ['']]
		string = np.array(string, dtype=float)
	except Exception as e: 
		print(e)

	return string

setInterface()