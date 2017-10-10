import tkinter as tk
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob

def setInterface():
	def clickTrain():
		###偵測列表選取的txt檔案   #################################listTxt如何取得的？
		selectionfile = listTxt.curselection()
		selectionfile = listTxt.get(selectionfile)
		print(selectionfile)
		array = readFile(selectionfile)
		###
		lrate=float(learnrateentry.get())
		ccondition=int(convergenceentry.get())
		print ("learnrate:", lrate)
		print ("convergence:", ccondition)
		train(array, ccondition, lrate)

	interface = tk.Tk()
	#創造視窗
	interface.title('interface')
	interface.geometry('500x500')
	#學習率字幕
	learnrate = tk.Label(interface, text="learnrate")
	learnrate.pack()
	#input learnrate
	learnrateentry = tk.Entry(interface)
	learnrateentry.insert(0, "0.5")
	learnrateentry.pack()
	#收斂字幕
	convergence = tk.Label(interface, text="convergence")
	convergence.pack()
	#input convergence
	convergenceentry = tk.Entry(interface)
	convergenceentry.insert(0, "10")
	convergenceentry.pack()
	#列出txt檔案
	listTxt = tk.Listbox(interface)
	haveTxt = glob.glob("*.txt")
	for txt in haveTxt:
		listTxt.insert(0, txt)
	# listTxt.bind('<<ListboxSelect>>', fileSelection)
	listTxt.pack()
	#訓練按鈕
	trainbtn = tk.Button(interface, text="train", command=clickTrain)
	trainbtn.pack()
	#讓視窗實現
	interface.mainloop()

def train(array, ccondition=100, lrate=0.5): 
	
	inputx, outputy, row, col = setInitialization(array)

	#選擇2/3的隨機訓練data
	trainDatasIndex = np.random.choice(inputx.shape[0], size=int(row*2/3) + 1, replace=False)
	trainDatas = inputx[trainDatasIndex, :]
	# print("trainDatas: ", trainDatas)
	#選擇1/3的隨機測試data
	testDatasIndex = np.arange(0, row)
	testDatasIndex = set(testDatasIndex) - set(trainDatasIndex)
	testDatasIndex = list(testDatasIndex)
	testDatas = inputx[testDatasIndex, :]
	# print("testDatas: ", testDatas)
	#正確“訓練辨識”數量
	#正確“測試辨識”數量
	trainIdnumber = 0
	testIdnumber = 0
	##神經元(perceptron)outputy數量 weight初始化
	y = np.zeros(int(np.amax(outputy))+1)
	weight = np.zeros(shape=(y.shape[0], col))
	for i in range(y.shape[0]):
		weight[i] = np.random.rand(1, col)
	##儲存最後一次outputy結果，用來畫出圖刑
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

	##########畫出圖形##########
	try:
		plt.figure(1, figsize=(8, 6))
		plt.subplot(221)
		showPlot(trainDatas, trainOutputResult)
		plt.subplot(222)
		showPlot(testDatas, testOutputResult)
		plt.show()
	except Exception as e:
		pass

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

def showPlot(Datas, outputResult):
	Datas = np.hsplit(Datas, [1])
	Datas = np.hsplit(Datas[1], [1])
	
	plt.scatter(Datas[0], Datas[1], c='red', label='perceptron1')
	plt.ylim((-5, 5))
	plt.legend()
	

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