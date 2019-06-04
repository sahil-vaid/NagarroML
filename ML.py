from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import scikitplot as skplt
import seaborn as sn
import pandas as pd






t = np.linspace(0,10,80)
mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Normal\\97")
nDE = []
nDE = list(mat['X097_DE_time'])
nMainDE = []
for j in nDE:
    for k in j:
        nMainDE.append(k)
nFE = list(mat['X097_FE_time'])
nMainFE = []
for j in nFE:
   for k in j:
       nMainFE.append(k)



def loadData(path, key):
	mat = loadmat(path)
	nDE = []
	list1 = []
	list1= list(mat[key])
	return list1

def plotMe(list1,list2):
	plt.plot(t,nMainDE[720:800],'r',label = 'Normal DE')
	plt.plot(t,list1[720:800],'b',label = 'Faulty DE')
	plt.plot(t,nMainFE[720:800],'g',label = 'Normal FE')
	plt.plot(t,list2[720:800],'y',label = 'Faulty FE')
	plt.title("Data Representation")
	plt.legend(loc = 'lower right')
	# plt.savefig("NormalDE(RED),IRDE10(Blue),NormalFE(Green),IRFE10(Yellow)")
	plt.show()
	plt.clf()
	plt.close()



##########################  Grade1 LOAD 0  #####################

irDE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\109", 'X109_DE_time')
irFE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\109", 'X109_FE_time')
bDE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\122",'X122_DE_time')
bFE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\122",'X122_FE_time')
urDE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\135",'X135_DE_time')
urFE10 = loadData("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\135",'X135_FE_time')



##########################  Grade1 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\110")
#print(mat.keys())
irDE11 = []
irDE11 = list(mat['X110_DE_time'])
#print(len(irDE11))

irFE11 = []
irFE11 = list(mat['X110_FE_time'])
#print(len(irFE11))



############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\123")
#print(mat.keys())
bDE11 = []
bDE11 = list(mat['X123_DE_time'])
#print(len(bDE11))

bFE11 = []
bFE11 = list(mat['X123_FE_time'])
#print(len(bFE11))



#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\136")
#print(mat.keys())
urDE11 = []
urDE11 = list(mat['X136_DE_time'])
#print(len(urDE11))

urFE11 = []
urFE11 = list(mat['X136_FE_time'])
#print(len(urFE11))



##########################  Grade1 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\111")
#print(mat.keys())
irDE12 = []
irDE12 = list(mat['X111_DE_time'])
#print(len(irDE12))

irFE12 = []
irFE12 = list(mat['X111_FE_time'])
#print(len(irFE12))

##plt.show()
##plt.savefig("NormalDE(RED),IRDE12(Blue),NormalFE(Green),IRFE12(Yellow)")


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\124")
#print(mat.keys())
bDE12 = []
bDE12 = list(mat['X124_DE_time'])
#print(len(bDE12))

bFE12 = []
bFE12 = list(mat['X124_FE_time'])
#print(len(bFE12))

##plt.savefig("NormalDE(RED),bDE12(Blue),NormalFE(Green),bFE12(Yellow)")
##plt.show()




#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\137")
#print(mat.keys())
urDE12 = []
urDE12 = list(mat['X137_DE_time'])
#print(len(urDE12))

urFE12 = []
urFE12 = list(mat['X137_FE_time'])
#print(len(urFE12))


##########################  Grade1 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\112")
#print(mat.keys())
irDE13 = []
irDE13 = list(mat['X112_DE_time'])
#print(len(irDE13))

irFE13 = []
irFE13 = list(mat['X112_FE_time'])
#print(len(irFE13))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\125")
#print(mat.keys())
bDE13 = []
bDE13 = list(mat['X125_DE_time'])
#print(len(bDE13))

bFE13 = []
bFE13 = list(mat['X125_FE_time'])
#print(len(bFE13))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\138")
#print(mat.keys())
urDE13 = []
urDE13 = list(mat['X138_DE_time'])
#print(len(urDE12))

urFE13 = []
urFE13 = list(mat['X138_FE_time'])
#print(len(urFE13))





##########################  Grade2 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\174")
#print(mat.keys())
irDE20 = []
irDE20 = list(mat['X173_DE_time'])
#print(len(irDE20))

irFE20 = []
irFE20 = list(mat['X173_FE_time'])
#print(len(irFE20))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\189")
#print(mat.keys())
bDE20 = []
bDE20 = list(mat['X189_DE_time'])
#print(len(bDE20))

bFE20 = []
bFE20 = list(mat['X189_FE_time'])
#print(len(bFE20))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\201")
#print(mat.keys())
urDE20 = []
urDE20 = list(mat['X201_DE_time'])
#print(len(urDE20))

urFE20 = []
urFE20 = list(mat['X201_FE_time'])
#print(len(urFE20))


##########################  Grade2 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\175")
#print(mat.keys())
irDE21 = []
irDE21 = list(mat['X175_DE_time'])
#print(len(irDE21))

irFE21 = []
irFE21 = list(mat['X175_FE_time'])
#print(len(irFE21))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\190")
#print(mat.keys())
bDE21 = []
bDE21 = list(mat['X190_DE_time'])
#print(len(bDE21))

bFE21 = []
bFE21 = list(mat['X190_FE_time'])
#print(len(bFE21))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\202")
#print(mat.keys())
urDE21 = []
urDE21 = list(mat['X202_DE_time'])
#print(len(urDE21))

urFE21 = []
urFE21 = list(mat['X202_FE_time'])
#print(len(urFE21))


##########################  Grade2 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\176")
#print(mat.keys())
irDE22 = []
irDE22 = list(mat['X176_DE_time'])
#print(len(irDE22))

irFE22 = []
irFE22 = list(mat['X176_FE_time'])
#print(len(irFE22))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\191")
#print(mat.keys())
bDE22 = []
bDE22 = list(mat['X191_DE_time'])
#print(len(bDE22))

bFE22 = []
bFE22 = list(mat['X191_FE_time'])
#print(len(bFE22))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\203")
#print(mat.keys())
urDE22 = []
urDE22 = list(mat['X203_DE_time'])
#print(len(urDE22))

urFE22 = []
urFE22 = list(mat['X203_FE_time'])
#print(len(urFE22))



##########################  Grade2 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\177")
#print(mat.keys())
irDE23 = []
irDE23 = list(mat['X177_DE_time'])
#print(len(irDE23))

irFE23 = []
irFE23 = list(mat['X177_FE_time'])
#print(len(irFE23))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\192")
#print(mat.keys())
bDE23 = []
bDE23 = list(mat['X192_DE_time'])
#print(len(bDE23))

bFE23 = []
bFE23 = list(mat['X192_FE_time'])
#print(len(bFE23))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\204")
#print(mat.keys())
urDE23 = []
urDE23 = list(mat['X204_DE_time'])
#print(len(urDE23))

urFE23 = []
urFE23 = list(mat['X204_FE_time'])
#print(len(urFE23))




##########################  Grade3 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\213")
#print(mat.keys())
irDE30 = []
irDE30 = list(mat['X213_DE_time'])
#print(len(irDE30))

irFE30 = []
irFE30 = list(mat['X213_FE_time'])
#print(len(irFE30))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\226")
#print(mat.keys())
bDE30 = []
bDE30 = list(mat['X226_DE_time'])
#print(len(bDE30))

bFE30 = []
bFE30 = list(mat['X226_FE_time'])
#print(len(bFE30))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\238")
#print(mat.keys())
urDE30 = []
urDE30 = list(mat['X238_DE_time'])
#print(len(urDE30))

urFE30 = []
urFE30 = list(mat['X238_FE_time'])
#print(len(urFE30))


##########################  Grade3 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\214")
#print(mat.keys())
irDE31 = []
irDE31 = list(mat['X214_DE_time'])
#print(len(irDE31))

irFE31 = []
irFE31 = list(mat['X214_FE_time'])
#print(len(irFE31))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\227")
#print(mat.keys())
bDE31 = []
bDE31 = list(mat['X227_DE_time'])
#print(len(bDE31))

bFE31 = []
bFE31 = list(mat['X227_FE_time'])
#print(len(bFE31))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\239")
#print(mat.keys())
urDE31 = []
urDE31 = list(mat['X239_DE_time'])
#print(len(urDE31))

urFE31 = []
urFE31 = list(mat['X239_FE_time'])
#print(len(urFE31))


##########################  Grade3 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\215")
#print(mat.keys())
irDE32 = []
irDE32 = list(mat['X215_DE_time'])
#print(len(irDE32))

irFE32 = []
irFE32 = list(mat['X215_FE_time'])
#print(len(irFE32))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\228")
#print(mat.keys())
bDE32 = []
bDE32 = list(mat['X228_DE_time'])
#print(len(bDE32))

bFE32 = []
bFE32 = list(mat['X228_FE_time'])
#print(len(bFE32))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\240")
#print(mat.keys())
urDE32 = []
urDE32 = list(mat['X240_DE_time'])
#print(len(urDE32))

urFE32 = []
urFE32 = list(mat['X240_FE_time'])
#print(len(urFE32))



##########################  Grade3 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\217")
#print(mat.keys())
irDE33 = []
irDE33 = list(mat['X217_DE_time'])
#print(len(irDE33))

irFE33 = []
irFE33 = list(mat['X217_FE_time'])
#print(len(irFE33))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\229")
#print(mat.keys())
bDE33 = []
bDE33 = list(mat['X229_DE_time'])
#print(len(bDE33))

bFE33 = []
bFE33 = list(mat['X229_FE_time'])
#print(len(bFE33))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\241")
#print(mat.keys())
urDE33 = []
urDE33 = list(mat['X241_DE_time'])
#print(len(urDE33))

urFE33 = []
urFE33 = list(mat['X241_FE_time'])
#print(len(urFE33))


#########################PLOTTING ALL DATA##########################################

# plotMe(irDE10,irFE10)
# plotMe(bDE10,bFE10)
# plotMe(urDE10,urFE10)
# plotMe(irDE11,irFE11)
# plotMe(bDE11,bFE11)
# plotMe(urDE11,urFE11)
# plotMe(irDE12,irFE12)
# plotMe(bDE12,bFE12)
# plotMe(urDE12,urFE12)
# plotMe(irDE13,irFE13)
# plotMe(bDE13,bFE13)
# plotMe(urDE13,urFE13)

# plotMe(irDE20,irFE20)
# plotMe(bDE20,bFE20)
# plotMe(urDE20,urFE20)
# plotMe(irDE21,irFE21)
# plotMe(bDE21,bFE21)
# plotMe(urDE21,urFE21)
# plotMe(irDE22,irFE22)
# plotMe(bDE22,bFE22)
# plotMe(urDE22,urFE22)
# plotMe(irDE23,irFE23)
# plotMe(bDE23,bFE23)
# plotMe(urDE23,urFE23)

# plotMe(irDE30,irFE30)
# plotMe(bDE30,bFE30)
# plotMe(urDE30,urFE30)
# plotMe(irDE31,irFE31)
# plotMe(bDE31,bFE31)
# plotMe(urDE31,urFE31)
# plotMe(irDE32,irFE32)
# plotMe(bDE32,bFE32)
# plotMe(urDE32,urFE32)
# plotMe(irDE33,irFE33)
# plotMe(bDE33,bFE33)
# plotMe(urDE33,urFE33)






''' FOR PLOTS 25 SUM  '''

newMainDE = []
count = 0
while(count + 30 < len(nMainDE)):
	newMainDE.append(sum(nMainDE[count:(count+30)]))
	count += 30
#print("forplots ", len(newMainDE))

newIrDe10 = []

count = 0
while(count + 30 < len(irDE10)):
	newIrDe10.append(sum(irDE10[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe10))

newIrDe11 = []

count = 0
while(count + 30 < len(irDE11)):
	newIrDe11.append(sum(irDE11[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe11))

newIrDe12 = []

count = 0
while(count + 30 < len(irDE12)):
	newIrDe12.append(sum(irDE12[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe12))

newIrDe13 = []

count = 0
while(count + 30 < len(irDE13)):
	newIrDe13.append(sum(irDE13[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe13))

newIrDe20 = []

count = 0
while(count + 30 < len(irDE20)):
	newIrDe20.append(sum(irDE20[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe20))

newIrDe21 = []

count = 0
while(count + 30 < len(irDE21)):
	newIrDe21.append(sum(irDE21[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe21))

newIrDe22 = []

count = 0
while(count + 30 < len(irDE22)):
	newIrDe22.append(sum(irDE22[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe22))

newIrDe23 = []

count = 0
while(count + 30 < len(irDE23)):
	newIrDe23.append(sum(irDE23[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe23))

newIrDe30 = []

count = 0
while(count + 30 < len(irDE30)):
	newIrDe30.append(sum(irDE30[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe30))

newIrDe31 = []

count = 0
while(count + 30 < len(irDE31)):
	newIrDe31.append(sum(irDE31[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe31))

newIrDe32 = []

count = 0
while(count + 30 < len(irDE32)):
	newIrDe32.append(sum(irDE32[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe32))

newIrDe33 = []

count = 0
while(count + 30 < len(irDE33)):
	newIrDe33.append(sum(irDE33[count:(count+30)]))
	count += 30
#print("forplots ",len(newIrDe33))



newurDe10 = []

count = 0
while(count + 30 < len(urDE10)):
	newurDe10.append(sum(urDE10[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe10))

newurDe11 = []

count = 0
while(count + 30 < len(urDE11)):
	newurDe11.append(sum(urDE11[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe11))

newurDe12 = []

count = 0
while(count + 30 < len(urDE12)):
	newurDe12.append(sum(urDE12[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe12))

newurDe13 = []

count = 0
while(count + 30 < len(urDE13)):
	newurDe13.append(sum(urDE13[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe13))

newurDe20 = []

count = 0
while(count + 30 < len(urDE20)):
	newurDe20.append(sum(urDE20[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe20))

newurDe21 = []

count = 0
while(count + 30 < len(urDE21)):
	newurDe21.append(sum(urDE21[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe21))

newurDe22 = []

count = 0
while(count + 30 < len(urDE22)):
	newurDe22.append(sum(urDE22[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe22))

newurDe23 = []

count = 0
while(count + 30 < len(urDE23)):
	newurDe23.append(sum(urDE23[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe23))

newurDe30 = []

count = 0
while(count + 30 < len(urDE30)):
	newurDe30.append(sum(urDE30[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe30))

newurDe31 = []

count = 0
while(count + 30 < len(urDE31)):
	newurDe31.append(sum(urDE31[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe31))

newurDe32 = []

count = 0
while(count + 30 < len(urDE32)):
	newurDe32.append(sum(urDE32[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe32))

newurDe33 = []

count = 0
while(count + 30 < len(urDE33)):
	newurDe33.append(sum(urDE33[count:(count+30)]))
	count += 30
#print("forplots ",len(newurDe33))



newbDe10 = []

count = 0
while(count + 30 < len(bDE10)):
	newbDe10.append(sum(bDE10[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe10))

newbDe11 = []

count = 0
while(count + 30 < len(bDE11)):
	newbDe11.append(sum(bDE11[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe11))

newbDe12 = []

count = 0
while(count + 30 < len(bDE12)):
	newbDe12.append(sum(bDE12[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe12))

newbDe13 = []

count = 0
while(count + 30 < len(bDE13)):
	newbDe13.append(sum(bDE13[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe13))

newbDe20 = []

count = 0
while(count + 30 < len(bDE20)):
	newbDe20.append(sum(bDE20[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe20))

newbDe21 = []

count = 0
while(count + 30 < len(bDE21)):
	newbDe21.append(sum(bDE21[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe21))

newbDe22 = []

count = 0
while(count + 30 < len(bDE22)):
	newbDe22.append(sum(bDE22[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe22))

newbDe23 = []

count = 0
while(count + 30 < len(bDE23)):
	newbDe23.append(sum(bDE23[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe23))

newbDe30 = []

count = 0
while(count + 30 < len(bDE30)):
	newbDe30.append(sum(bDE30[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe30))

newbDe31 = []

count = 0
while(count + 30 < len(bDE31)):
	newbDe31.append(sum(bDE31[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe31))

newbDe32 = []

count = 0
while(count + 30 < len(bDE32)):
	newbDe32.append(sum(bDE32[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe32))

newbDe33 = []

count = 0
while(count + 30 < len(bDE33)):
	newbDe33.append(sum(bDE33[count:(count+30)]))
	count += 30
#print("forplots ",len(newbDe33))

t = np.linspace(0,1,100)

plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe10[:100],'b')

plt.title("SumDENormal vs SumDeIR10")
# plt.savefig("SumDENormal vs SumDeIR10")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe11[:100],'b')

plt.title("SumDENormal vs SumDeIR11")
# plt.savefig("SumDENormal vs SumDeIR11")
# plt.show()



plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe12[:100],'b')

plt.title("SumDENormal vs SumDeIR12")
# plt.savefig("SumDENormal vs SumDeIR12")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe13[:100],'b')

plt.title("SumDENormal vs SumDeIR13")
# plt.savefig("SumDENormal vs SumDeIR13")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe20[:100],'b')

plt.title("SumDENormal vs SumDeIR20")
# plt.savefig("SumDENormal vs SumDeIR20")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe21[:100],'b')

plt.title("SumDENormal vs SumDeIR21")
# plt.savefig("SumDENormal vs SumDeIR21")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe22[:100],'b')

plt.title("SumDENormal vs SumDeIR22")
# plt.savefig("SumDENormal vs SumDeIR22")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe23[:100],'b')

plt.title("SumDENormal vs SumDeIR23")
# plt.savefig("SumDENormal vs SumDeIR23")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe30[:100],'b')

plt.title("SumDENormal vs SumDeIR30")
# plt.savefig("SumDENormal vs SumDeIR30")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe31[:100],'b')

plt.title("SumDENormal vs SumDeIR31")
# plt.savefig("SumDENormal vs SumDeIR31")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe32[:100],'b')

plt.title("SumDENormal vs SumDeIR32")
# plt.savefig("SumDENormal vs SumDeIR32")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newIrDe33[:100],'b')

plt.title("SumDENormal vs SumDeIR33")
# plt.savefig("SumDENormal vs SumDeIR33")
# plt.show()




plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe10[:100],'b')

plt.title("SumDENormal vs SumDeur10")
# plt.savefig("SumDENormal vs SumDeuR10")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe11[:100],'b')

plt.title("SumDENormal vs SumDeuR11")
# plt.savefig("SumDENormal vs SumDeuR11")
# plt.show()



plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe12[:100],'b')

plt.title("SumDENormal vs SumDeuR12")
# plt.savefig("SumDENormal vs SumDeuR12")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe13[:100],'b')

plt.title("SumDENormal vs SumDeuR13")
# plt.savefig("SumDENormal vs SumDeuR13")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe20[:100],'b')

plt.title("SumDENormal vs SumDeuR20")
# plt.savefig("SumDENormal vs SumDeuR20")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe21[:100],'b')

plt.title("SumDENormal vs SumDeuR21")
# plt.savefig("SumDENormal vs SumDeuR21")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe22[:100],'b')

plt.title("SumDENormal vs SumDeuR22")
# plt.savefig("SumDENormal vs SumDeuR22")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe23[:100],'b')

plt.title("SumDENormal vs SumDeuR23")
# plt.savefig("SumDENormal vs SumDeuR23")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe30[:100],'b')

plt.title("SumDENormal vs SumDeuR30")
# plt.savefig("SumDENormal vs SumDeuR30")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe31[:100],'b')

plt.title("SumDENormal vs SumDeuR31")
# plt.savefig("SumDENormal vs SumDeuR31")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe32[:100],'b')

plt.title("SumDENormal vs SumDeuR32")
# plt.savefig("SumDENormal vs SumDeuR32")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newurDe33[:100],'b')

plt.title("SumDENormal vs SumDeuR33")
# plt.savefig("SumDENormal vs SumDeuR33")
# plt.show()






plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe10[:100],'b')

plt.title("SumDENormal vs SumDeb10")
# plt.savefig("SumDENormal vs SumDeb10")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe11[:100],'b')

plt.title("SumDENormal vs SumDeb11")
# plt.savefig("SumDENormal vs SumDeb11")
# plt.show()



plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe12[:100],'b')

plt.title("SumDENormal vs SumDeb12")
# plt.savefig("SumDENormal vs SumDeb12")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe13[:100],'b')

plt.title("SumDENormal vs SumDeb13")
# plt.savefig("SumDENormal vs SumDeb13")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe20[:100],'b')

plt.title("SumDENormal vs SumDeb20")
# plt.savefig("SumDENormal vs SumDeb20")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe21[:100],'b')

plt.title("SumDENormal vs SumDeb21")
# plt.savefig("SumDENormal vs SumDeb21")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe22[:100],'b')

plt.title("SumDENormal vs SumDeb22")
# plt.savefig("SumDENormal vs SumDeb22")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe23[:100],'b')

plt.title("SumDENormal vs SumDeb23")
# plt.savefig("SumDENormal vs SumDeb23")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe30[:100],'b')

plt.title("SumDENormal vs SumDeb30")
# plt.savefig("SumDENormal vs SumDeb30")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe31[:100],'b')

plt.title("SumDENormal vs SumDeb31")
# plt.savefig("SumDENormal vs SumDeb31")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe32[:100],'b')

plt.title("SumDENormal vs SumDeb32")
# plt.savefig("SumDENormal vs SumDeb32")
# plt.show()


plt.plot(t,newMainDE[:100],'r')
plt.plot(t,newbDe33[:100],'b')

plt.title("SumDENormal vs SumDeb33")
# plt.savefig("SumDENormal vs SumDeb33")
# plt.show()

plt.close()







'''''''''''''''''''''END PLOTS'''''''''''''''''''''''''''''''''''''''

def splitData(irDataDE, irDataFE, urDataDE, urDataFE, bDataDE, bDataFE):
	xDE = []
	for i in nMainDE:
		xDE.append(i)
	for i in irDataDE:
		xDE.append(i)
	for i in urDataDE:
		xDE.append(i)
	for i in bDataDE:
		xDE.append(i)
	xFE = []
	for i in nMainFE:
		xFE.append(i)
	for i in irDataFE:
		xFE.append(i)
	for i in urDataFE:
		xFE.append(i)
	for i in bDataFE:
		xFE.append(i)



	X = np.vstack((xDE,xFE))

	Y =[]
	for i in nMainDE:
		Y.append(0)
	for i in irDataDE:
		Y.append(1)
	for i in urDataDE:
		Y.append(2)
	for i in bDataDE:
		Y.append(3)

	return train_test_split(X.T, Y, test_size=0.33, random_state=42)


finalList=[]
testList =[]
def Values(pRF,pKN, pDT,Ytest):
	tp =0
	tn =0
	fp =0
	fn =0

	for i in range(len(Ytest)):
		zeroCount =0
		oneCount = 0
		final = 0
		if(round(pRF[i]) == 0):
			zeroCount+=1
		else:
			oneCount+=1
		if(round(pKN[i]) == 0):
			zeroCount+=1
		else:
			oneCount+=1
		if(round(pDT[i]) == 0):
			zeroCount+=1
		else:
			oneCount+=1
		if(zeroCount > oneCount):
			final = 0
			finalList.append(0)
		else:
			final = 1
			finalList.append(1)

		
		if(final == 1 and Ytest == 0):
			print("Test ---> ", Ytest[i], "RF --> ",pRF[i], "DT --> " , pDT[i], "KN --> ",pKN[i] )
		

		if(final == 1 and (Ytest[i] == 1 or Ytest[i] == 2 or Ytest[i] == 3)):
			tn +=1
		elif(final == 0 and Ytest[i] == 0):
			tp += 1
		elif(final == 0 and (Ytest[i] == 1 or Ytest[i] == 2 or Ytest[i] == 3)):
			fp +=1
		elif(final == 1 and Ytest==0):
			fn +=1

		if(Ytest[i] == 1 or Ytest[i] == 2 or Ytest[i] == 3):
			testList.append(1)
		else:
			testList.append(0)


	return tp, tn, fp,fn


def drawGraphs(predicted, given):

	fpr, tpr, threshold = metrics.roc_curve(given,predicted)
	roc_auc = metrics.auc(fpr, tpr)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([-0.05, 1.1])
	plt.ylim([-0.05, 1.1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.show()




print(''''Training testing for single class''')



nMainDETrain = nMainDE[ :(len(nMainDE)//3) ]
nMainDETest = nMainDE
nMainFETrain = nMainFE[ : len(nMainFE)//3 ]
nMainFETest = nMainFE


def trainTest(DE,FE):
	trainListD = DE[ : (len(irFE10)//3)]
	testListD = DE
	trainListF = FE[ : (len(irFE10)//3)]
	testListF = FE
	return trainListD,testListD,trainListF,testListF

def XYreturn(DEtrain,DEtest,FEtrain,FEtest):
	XtrainDE = []
	for j in nMainDETrain:
	    XtrainDE.append(j)
	for j in DEtrain:
	    XtrainDE.append(j)

	YtrainDE = []
	for i in nMainDETrain:
	    YtrainDE.append(0)
	for i in DEtrain:
	    YtrainDE.append(1)
	#print(len(Ytrain))
	##Ytrain = np.asarray(Ytrain).reshape(-1,1)

	XtestDE = []
	for j in nMainDETest:
	    XtestDE.append(j)
	for j in DEtest:
	    XtestDE.append(j)


	YtestDE =[]
	for i in nMainDETest:
	    YtestDE.append(0)
	for i in DEtest:
	    YtestDE.append(1)

	XtrainFE = []
	for j in nMainFETrain:
	    XtrainFE.append(j)
	for j in FEtrain:
	    XtrainFE.append(j)

	YtrainFE = []
	for i in nMainFETrain:
	    YtrainFE.append(0)
	for i in FEtrain:
	    YtrainFE.append(1)
	##Ytrain = np.asarray(Ytrain).reshape(-1,1)

	XtestFE = []
	for j in nMainFETest:
	    XtestFE.append(j)
	for j in FEtest:
	    XtestFE.append(j)


	YtestFE =[]
	for i in nMainFETest:
	    YtestFE.append(0)
	for i in FEtest:
	    YtestFE.append(1)

	Xtest = np.vstack((XtestDE,XtestFE))
	Xtrain = np.vstack((XtrainDE,XtrainFE))
	#print(np.shape(Xtest),np.shape(Xtrain))


	Ytest =[]
	for i in nMainFETest:
	    Ytest.append(0)
	for i in FEtest:
	    Ytest.append(1)

	Ytrain = []
	for i in nMainFETrain:
		Ytrain.append(0)
	for i in FEtrain:
		Ytrain.append(1)

	return Xtest,Xtrain,Ytest,Ytrain

def predictRF(Xtest,Xtrain,Ytest,Ytrain):

	rf = RandomForestRegressor()
	rf.fit(Xtrain.T,Ytrain)
	predictions = rf.predict(Xtest.T)
	accuracy = accuracy_score(predictions.round(),Ytest)
	print("RandomForest DE FE  ", accuracy)
	return predictions

def predictLR(Xtest,Xtrain,Ytest,Ytrain):

	LR = LinearRegression().fit(Xtrain.T,Ytrain)
	predictions=LR.predict(Xtest.T)
	accuracy = accuracy_score(predictions.round(),Ytest)
	print("LinearRegression DE FE ", accuracy)
	plt.close()
	plt.figure()
	tx=Xtrain.T
	plt.scatter(tx[:1000,0],tx[:1000,1],c=Ytrain[:1000])
	plt.show()
	return predictions

def predictKNN(Xtest,Xtrain,Ytest,Ytrain):
	
	neigh = KNeighborsClassifier()
	neigh.fit(Xtrain.T,Ytrain)
	predictions=neigh.predict(Xtest.T)
	accuracy = accuracy_score(predictions.round(),Ytest)
	print("KNN DE FE ", accuracy)
	return predictions


def predictSVM(Xtest,Xtrain,Ytest,Ytrain):

	clf = svm.SVC(gamma='scale')
	clf.fit(Xtrain.T,Ytrain)
	predictions = clf.predict(Xtest.T)
	accuracy = accuracy_score(predictions.round(),Ytest)
	print("SVM DE FE B", accuracy)	
	return predictions

def predictDTree(Xtest,Xtrain,Ytest,Ytrain):

	dt = DecisionTreeClassifier(random_state=0).fit(Xtrain,Ytrain)
	predictions = dt.predict(Xtest)
	accuracy = accuracy_score(predictions.round(),Ytest)
	print("Decision Tree DE FE ", accuracy)
	return predictions


def plotMat(matrix):

	plt.matshow(matrix)
	plt.title('Confusion Matrix')
	plt.text(0 , 0 , matrix[0][0] , va= 'center', ha = 'center')
	plt.text(0 , 1 , matrix[0][1] , va= 'center', ha = 'center')
	plt.text(1 , 0 , matrix[1][0] , va= 'center', ha = 'center')
	plt.text(1 , 1 , matrix[1][1] , va= 'center', ha = 'center')
	plt.colorbar()
	plt.show()
	plt.clf()
	plt.close()




print("MultiClass Testing")

print("Grade 1 label 0")

Xtrain, Xtest, Ytrain, Ytest = splitData(irDE10, irFE10 , urDE10 , urFE10, bDE10, bFE10)
pRF = predictRF(Xtest.T,Xtrain.T,Ytest,Ytrain)
pKN = predictKNN(Xtest.T,Xtrain.T,Ytest,Ytrain)
pDT = predictDTree(Xtest,Xtrain,Ytest,Ytrain)
tp, tn, fp,fn = Values(pRF,pKN, pDT,Ytest)

print(tp,tn,fp,fn)

Accuracy = (tp+tn)/(tp+fp+fn+tn)
print("Final Accuracy ", Accuracy)

Precision = tp/(tp+fp)
print("Final precision ", Precision)

Recall = tp/(tp+fn)
print("Final recall", Recall)
plt.clf()
plt.close()
drawGraphs(finalList,testList)
plt.close()

print("Grade 2 label 1")

Xtrain, Xtest, Ytrain, Ytest = splitData(irDE21, irFE21 , urDE21 , urFE21, bDE21, bFE21)
pRF = predictRF(Xtest.T,Xtrain.T,Ytest,Ytrain)
pKN = predictKNN(Xtest.T,Xtrain.T,Ytest,Ytrain)
pDT = predictDTree(Xtest,Xtrain,Ytest,Ytrain)
tp, tn, fp,fn = Values(pRF,pKN, pDT,Ytest)

print(tp,tn,fp,fn)

Accuracy = (tp+tn)/(tp+fp+fn+tn)
print("Final Accuracy ", Accuracy)

Precision = tp/(tp+fp)
print("Final precision ", Precision)

Recall = tp/(tp+fn)
print("Final recall", Recall)
plt.clf()
plt.close()
drawGraphs(finalList,testList)
plt.close()
accuracy = accuracy_score(finalList,testList)
print("check ", accuracy)

	
print('''-------------------------------------------- GLOBAL1------------------------------------------------''')

# ##################### Train  IR DE FE ################################


irDE10Train,irDE10Test,irFE10Train,irFE10Test = trainTest(irDE10,irFE10)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE10Train,irDE10Test,irFE10Train,irFE10Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)



irDE11Train,irDE11Test,irFE11Train,irFE11Test = trainTest(irDE11,irFE11)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE11Train,irDE11Test,irFE11Train,irFE11Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


irDE12Train,irDE12Test,irFE12Train,irFE12Test = trainTest(irDE12,irFE12)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE12Train,irDE12Test,irFE12Train,irFE12Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


irDE13Train,irDE13Test,irFE13Train,irFE13Test = trainTest(irDE13,irFE13)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE13Train,irDE13Test,irFE13Train,irFE13Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)

# ##################### Train  UR DE FE ################################


urDE10Train,urDE10Test,urFE10Train,urFE10Test = trainTest(urDE10,urFE10)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE10Train,urDE10Test,urFE10Train,urFE10Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)



urDE11Train,urDE11Test,urFE11Train,urFE11Test = trainTest(urDE11,urFE11)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE11Train,urDE11Test,urFE11Train,urFE11Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


urDE12Train,urDE12Test,urFE12Train,urFE12Test = trainTest(urDE12,urFE12)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE12Train,urDE12Test,urFE12Train,urFE12Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


urDE13Train,urDE13Test,urFE13Train,urFE13Test = trainTest(urDE13,urFE13)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE13Train,irDE13Test,irFE13Train,irFE13Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


print('''-------------------------------------------- GLOBAL2------------------------------------------------''')

# ##################### Train  IR DE FE ################################


irDE20Train,irDE20Test,irFE20Train,irFE20Test = trainTest(irDE20,irFE20)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE20Train,irDE20Test,irFE20Train,irFE20Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)



irDE21Train,irDE21Test,irFE21Train,irFE21Test = trainTest(irDE21,irFE21)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE21Train,irDE21Test,irFE21Train,irFE21Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


irDE22Train,irDE22Test,irFE22Train,irFE22Test = trainTest(irDE22,irFE22)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE22Train,irDE22Test,irFE22Train,irFE22Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


irDE23Train,irDE23Test,irFE23Train,irFE23Test = trainTest(irDE23,irFE23)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE23Train,irDE23Test,irFE23Train,irFE23Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)

# ##################### Train  UR DE FE ################################


urDE20Train,urDE20Test,urFE20Train,urFE20Test = trainTest(urDE20,urFE20)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE20Train,urDE20Test,urFE20Train,urFE20Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)



urDE21Train,urDE21Test,urFE21Train,urFE21Test = trainTest(urDE21,urFE21)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE21Train,urDE21Test,urFE21Train,urFE21Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


urDE22Train,urDE22Test,urFE22Train,urFE22Test = trainTest(urDE22,urFE22)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(urDE22Train,urDE22Test,urFE22Train,urFE22Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)


urDE23Train,urDE23Test,urFE23Train,urFE23Test = trainTest(urDE23,urFE23)
Xtest,Xtrain,Ytest,Ytrain = XYreturn(irDE23Train,irDE23Test,irFE23Train,irFE23Test)
print('''DE FE PREDICTION FOR IR ONLY''')
predictions = predictRF(Xtest,Xtrain,Ytest,Ytrain)

matrix = metrics.confusion_matrix(Ytest,predictions.round())
plotMat(matrix)
drawGraphs(predictions,Ytest)



