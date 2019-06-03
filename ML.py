from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

t = np.linspace(0,5,100)

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Normal\\97")
print(mat.keys())
nDE = []
##nDE.append()
##print(len(nDE))

nDE = list(mat['X097_DE_time'])
##nMainDe = np.reshape(1,243938)
print(np.shape(nDE))
nMainDE = []
for j in nDE:
    for k in j:
        nMainDE.append(k)
print(len(nMainDE))

##nFE = []
##nFE.append()
##print(len(nFE))
nMainFE = list(mat['X097_FE_time'])
##for j in nFE:
##    for k in j:
##        nMainFE.append(k)
print(len(nMainFE))

##########################  Grade1 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\109")
print(mat.keys())
irDE10 = []
nDE = []
irDE10= list(mat['X109_DE_time'])


print(len(irDE10))

irFE10 = []
irFE10 = list(mat['X109_FE_time'])
print(len(irFE10))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE10[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE10[:100],'y')
plt.title("NormalDE(RED),IRDE10(Blue),NormalFE(Green),IRFE10(Yellow)")
##plt.savefig("NormalDE(RED),IRDE10(Blue),NormalFE(Green),IRFE10(Yellow)")
##plt.show()
plt.close()
############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\122")
print(mat.keys())
bDE10 = []
bDE10 = list(mat['X122_DE_time'])
print(len(bDE10))

bFE10 = []
bFE10 = list(mat['X122_FE_time'])
print(len(bFE10))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE10[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE10[:100],'y')
plt.title("NormalDE(RED),bDE10(Blue),NormalFE(Green),bFE10(Yellow)")
##plt.savefig("NormalDE(RED),bDE10(Blue),NormalFE(Green),bFE10(Yellow)")
##plt.show()
plt.close()

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\135")
print(mat.keys())
urDE10 = []
urDE10 = list(mat['X135_DE_time'])
print(len(urDE10))

urFE10 = []
urFE10 = list(mat['X135_FE_time'])
print(len(urFE10))


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE10[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE10[:100],'y')
plt.title("NormalDE(RED),urDE10(Blue),NormalFE(Green),urFE10(Yellow)")
##plt.savefig("NormalDE(RED),urDE10(Blue),NormalFE(Green),urFE10(Yellow)")
##plt.show()

##########################  Grade1 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\110")
print(mat.keys())
irDE11 = []
irDE11 = list(mat['X110_DE_time'])
print(len(irDE11))

irFE11 = []
irFE11 = list(mat['X110_FE_time'])
print(len(irFE11))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE11[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE11[:100],'y')
plt.title("NormalDE(RED),IRDE11(Blue),NormalFE(Green),IRFE11(Yellow)")
##plt.savefig("NormalDE(RED),IRDE11(Blue),NormalFE(Green),IRFE11(Yellow)")
##plt.show()

############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\123")
print(mat.keys())
bDE11 = []
bDE11 = list(mat['X123_DE_time'])
print(len(bDE11))

bFE11 = []
bFE11 = list(mat['X123_FE_time'])
print(len(bFE11))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE11[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE11[:100],'y')
plt.title("NormalDE(RED),bDE11(Blue),NormalFE(Green),bFE11(Yellow)")
##plt.savefig("NormalDE(RED),bDE11(Blue),NormalFE(Green),bFE11(Yellow)")
##plt.show()



#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\136")
print(mat.keys())
urDE11 = []
urDE11 = list(mat['X136_DE_time'])
print(len(urDE11))

urFE11 = []
urFE11 = list(mat['X136_FE_time'])
print(len(urFE11))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE11[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE11[:100],'y')
plt.title("NormalDE(RED),urDE11(Blue),NormalFE(Green),urFE11(Yellow)")
##plt.savefig("NormalDE(RED),urDE11(Blue),NormalFE(Green),urFE11(Yellow)")
##plt.show()


##########################  Grade1 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\111")
print(mat.keys())
irDE12 = []
irDE12 = list(mat['X111_DE_time'])
print(len(irDE12))

irFE12 = []
irFE12 = list(mat['X111_FE_time'])
print(len(irFE12))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE12[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE12[:100],'y')
plt.title("NormalDE(RED),IRDE12(Blue),NormalFE(Green),IRFE12(Yellow)")
##plt.show()
##plt.savefig("NormalDE(RED),IRDE12(Blue),NormalFE(Green),IRFE12(Yellow)")


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\124")
print(mat.keys())
bDE12 = []
bDE12 = list(mat['X124_DE_time'])
print(len(bDE12))

bFE12 = []
bFE12 = list(mat['X124_FE_time'])
print(len(bFE12))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE12[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE12[:100],'y')
plt.title("NormalDE(RED),bDE12(Blue),NormalFE(Green),bFE12(Yellow)")
##plt.savefig("NormalDE(RED),bDE12(Blue),NormalFE(Green),bFE12(Yellow)")
##plt.show()




#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\137")
print(mat.keys())
urDE12 = []
urDE12 = list(mat['X137_DE_time'])
print(len(urDE12))

urFE12 = []
urFE12 = list(mat['X137_FE_time'])
print(len(urFE12))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE12[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE12[:100],'y')
plt.title("NormalDE(RED),urDE12(Blue),NormalFE(Green),urFE12(Yellow)")
##plt.savefig("NormalDE(RED),urDE12(Blue),NormalFE(Green),urFE12(Yellow)")
##plt.show()


##########################  Grade1 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\112")
print(mat.keys())
irDE13 = []
irDE13 = list(mat['X112_DE_time'])
print(len(irDE13))

irFE13 = []
irFE13 = list(mat['X112_FE_time'])
print(len(irFE13))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\125")
print(mat.keys())
bDE13 = []
bDE13 = list(mat['X125_DE_time'])
print(len(bDE13))

bFE13 = []
bFE13 = list(mat['X125_FE_time'])
print(len(bFE13))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load3\\138")
print(mat.keys())
urDE13 = []
urDE13 = list(mat['X138_DE_time'])
print(len(urDE12))

urFE13 = []
urFE13 = list(mat['X138_FE_time'])
print(len(urFE13))


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE13[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE13[:100],'y')
plt.title("NormalDE(RED),IRDE13(Blue),NormalFE(Green),IRFE13(Yellow)")
##plt.savefig("NormalDE(RED),IRDE13(Blue),NormalFE(Green),IRFE13(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE13[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE13[:100],'y')
plt.title("NormalDE(RED),bDE13(Blue),NormalFE(Green),bFE13(Yellow)")
##plt.savefig("NormalDE(RED),bDE13(Blue),NormalFE(Green),bFE13(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE13[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE13[:100],'y')
plt.title("NormalDE(RED),urDE13(Blue),NormalFE(Green),urFE13(Yellow)")
##plt.savefig("NormalDE(RED),urDE13(Blue),NormalFE(Green),urFE13(Yellow)")
##plt.show()




##########################  Grade2 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\174")
print(mat.keys())
irDE20 = []
irDE20 = list(mat['X173_DE_time'])
print(len(irDE20))

irFE20 = []
irFE20 = list(mat['X173_FE_time'])
print(len(irFE20))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\189")
print(mat.keys())
bDE20 = []
bDE20 = list(mat['X189_DE_time'])
print(len(bDE20))

bFE20 = []
bFE20 = list(mat['X189_FE_time'])
print(len(bFE20))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load0\\201")
print(mat.keys())
urDE20 = []
urDE20 = list(mat['X201_DE_time'])
print(len(urDE20))

urFE20 = []
urFE20 = list(mat['X201_FE_time'])
print(len(urFE20))


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE20[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE20[:100],'y')
plt.title("NormalDE(RED),IRDE20(Blue),NormalFE(Green),IRFE20(Yellow)")
##plt.savefig("NormalDE(RED),IRDE20(Blue),NormalFE(Green),IRFE20(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE20[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE20[:100],'y')
plt.title("NormalDE(RED),bDE20(Blue),NormalFE(Green),bFE20(Yellow)")
##plt.savefig("NormalDE(RED),bDE20(Blue),NormalFE(Green),bFE20(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE20[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE20[:100],'y')
plt.title("NormalDE(RED),urDE20(Blue),NormalFE(Green),urFE20(Yellow)")
##plt.savefig("NormalDE(RED),urDE20(Blue),NormalFE(Green),urFE20(Yellow)")
##plt.show()


##########################  Grade2 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\175")
print(mat.keys())
irDE21 = []
irDE21 = list(mat['X175_DE_time'])
print(len(irDE21))

irFE21 = []
irFE21 = list(mat['X175_FE_time'])
print(len(irFE21))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\190")
print(mat.keys())
bDE21 = []
bDE21 = list(mat['X190_DE_time'])
print(len(bDE21))

bFE21 = []
bFE21 = list(mat['X190_FE_time'])
print(len(bFE21))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load1\\202")
print(mat.keys())
urDE21 = []
urDE21 = list(mat['X202_DE_time'])
print(len(urDE21))

urFE21 = []
urFE21 = list(mat['X202_FE_time'])
print(len(urFE21))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE21[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE21[:100],'y')
plt.title("NormalDE(RED),IRDE21(Blue),NormalFE(Green),IRFE21(Yellow)")
##plt.savefig("NormalDE(RED),IRDE21(Blue),NormalFE(Green),IRFE21(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE21[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE21[:100],'y')
plt.title("NormalDE(RED),bDE21(Blue),NormalFE(Green),bFE21(Yellow)")
##plt.savefig("NormalDE(RED),bDE21(Blue),NormalFE(Green),bFE21(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE21[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE21[:100],'y')
plt.title("NormalDE(RED),urDE21(Blue),NormalFE(Green),urFE21(Yellow)")
##plt.savefig("NormalDE(RED),urDE21(Blue),NormalFE(Green),urFE21(Yellow)")
##plt.show()





##########################  Grade2 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\176")
print(mat.keys())
irDE22 = []
irDE22 = list(mat['X176_DE_time'])
print(len(irDE22))

irFE22 = []
irFE22 = list(mat['X176_FE_time'])
print(len(irFE22))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\191")
print(mat.keys())
bDE22 = []
bDE22 = list(mat['X191_DE_time'])
print(len(bDE22))

bFE22 = []
bFE22 = list(mat['X191_FE_time'])
print(len(bFE22))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load2\\203")
print(mat.keys())
urDE22 = []
urDE22 = list(mat['X203_DE_time'])
print(len(urDE22))

urFE22 = []
urFE22 = list(mat['X203_FE_time'])
print(len(urFE22))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE22[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE22[:100],'y')
plt.title("NormalDE(RED),IRDE22(Blue),NormalFE(Green),IRFE22(Yellow)")
##plt.savefig("NormalDE(RED),IRDE22(Blue),NormalFE(Green),IRFE22(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE22[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE22[:100],'y')
plt.title("NormalDE(RED),bDE22(Blue),NormalFE(Green),bFE22(Yellow)")
##plt.savefig("NormalDE(RED),bDE22(Blue),NormalFE(Green),bFE22(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE22[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE22[:100],'y')
plt.title("NormalDE(RED),urDE22(Blue),NormalFE(Green),urFE22(Yellow)")
##plt.savefig("NormalDE(RED),urDE22(Blue),NormalFE(Green),urFE22(Yellow)")
##plt.show()





##########################  Grade2 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\177")
print(mat.keys())
irDE23 = []
irDE23 = list(mat['X177_DE_time'])
print(len(irDE23))

irFE23 = []
irFE23 = list(mat['X177_FE_time'])
print(len(irFE23))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\192")
print(mat.keys())
bDE23 = []
bDE23 = list(mat['X192_DE_time'])
print(len(bDE23))

bFE23 = []
bFE23 = list(mat['X192_FE_time'])
print(len(bFE23))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade2\\load3\\204")
print(mat.keys())
urDE23 = []
urDE23 = list(mat['X204_DE_time'])
print(len(urDE23))

urFE23 = []
urFE23 = list(mat['X204_FE_time'])
print(len(urFE23))


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE23[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE23[:100],'y')
plt.title("NormalDE(RED),IRDE23(Blue),NormalFE(Green),IRFE23(Yellow)")
##plt.savefig("NormalDE(RED),IRDE23(Blue),NormalFE(Green),IRFE23(Yellow)")
##plt.show()



plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE23[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE23[:100],'y')
plt.title("NormalDE(RED),bDE23(Blue),NormalFE(Green),bFE23(Yellow)")
##plt.savefig("NormalDE(RED),bDE23(Blue),NormalFE(Green),bFE23(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE23[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE23[:100],'y')
plt.title("NormalDE(RED),urDE23(Blue),NormalFE(Green),urFE23(Yellow)")
##plt.savefig("NormalDE(RED),urDE22(Blue),NormalFE(Green),urFE22(Yellow)")
##plt.show()






##########################  Grade3 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\213")
print(mat.keys())
irDE30 = []
irDE30 = list(mat['X213_DE_time'])
print(len(irDE30))

irFE30 = []
irFE30 = list(mat['X213_FE_time'])
print(len(irFE30))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\226")
print(mat.keys())
bDE30 = []
bDE30 = list(mat['X226_DE_time'])
print(len(bDE30))

bFE30 = []
bFE30 = list(mat['X226_FE_time'])
print(len(bFE30))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load0\\238")
print(mat.keys())
urDE30 = []
urDE30 = list(mat['X238_DE_time'])
print(len(urDE30))

urFE30 = []
urFE30 = list(mat['X238_FE_time'])
print(len(urFE30))

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE30[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE30[:100],'y')
plt.title("NormalDE(RED),IRDE30(Blue),NormalFE(Green),IRFE30(Yellow)")
##plt.savefig("NormalDE(RED),IRDE30(Blue),NormalFE(Green),IRFE30(Yellow)")
##plt.show()



plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE30[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE30[:100],'y')
plt.title("NormalDE(RED),bDE30(Blue),NormalFE(Green),bFE30(Yellow)")
##plt.savefig("NormalDE(RED),bDE30(Blue),NormalFE(Green),bFE30(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE30[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE30[:100],'y')
plt.title("NormalDE(RED),urDE30(Blue),NormalFE(Green),urFE30(Yellow)")
##plt.savefig("NormalDE(RED),urDE30(Blue),NormalFE(Green),urFE30(Yellow)")
##plt.show()



##########################  Grade3 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\214")
print(mat.keys())
irDE31 = []
irDE31 = list(mat['X214_DE_time'])
print(len(irDE31))

irFE31 = []
irFE31 = list(mat['X214_FE_time'])
print(len(irFE31))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\227")
print(mat.keys())
bDE31 = []
bDE31 = list(mat['X227_DE_time'])
print(len(bDE31))

bFE31 = []
bFE31 = list(mat['X227_FE_time'])
print(len(bFE31))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load1\\239")
print(mat.keys())
urDE31 = []
urDE31 = list(mat['X239_DE_time'])
print(len(urDE31))

urFE31 = []
urFE31 = list(mat['X239_FE_time'])
print(len(urFE31))


##########################  Grade3 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\215")
print(mat.keys())
irDE32 = []
irDE32 = list(mat['X215_DE_time'])
print(len(irDE32))

irFE32 = []
irFE32 = list(mat['X215_FE_time'])
print(len(irFE32))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\228")
print(mat.keys())
bDE32 = []
bDE32 = list(mat['X228_DE_time'])
print(len(bDE32))

bFE32 = []
bFE32 = list(mat['X228_FE_time'])
print(len(bFE32))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load2\\240")
print(mat.keys())
urDE32 = []
urDE32 = list(mat['X240_DE_time'])
print(len(urDE32))

urFE32 = []
urFE32 = list(mat['X240_FE_time'])
print(len(urFE32))



##########################  Grade3 LOAD 3  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\217")
print(mat.keys())
irDE33 = []
irDE33 = list(mat['X217_DE_time'])
print(len(irDE33))

irFE33 = []
irFE33 = list(mat['X217_FE_time'])
print(len(irFE33))


################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\229")
print(mat.keys())
bDE33 = []
bDE33 = list(mat['X229_DE_time'])
print(len(bDE33))

bFE33 = []
bFE33 = list(mat['X229_FE_time'])
print(len(bFE33))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade3\\load3\\241")
print(mat.keys())
urDE33 = []
urDE33 = list(mat['X241_DE_time'])
print(len(urDE33))

urFE33 = []
urFE33 = list(mat['X241_FE_time'])
print(len(urFE33))


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE31[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE31[:100],'y')
plt.title("NormalDE(RED),IRDE31(Blue),NormalFE(Green),IRFE31(Yellow)")
##plt.savefig("NormalDE(RED),IRDE31(Blue),NormalFE(Green),IRFE31(Yellow)")
##plt.show()



plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE31[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE31[:100],'y')
plt.title("NormalDE(RED),bDE31(Blue),NormalFE(Green),bFE31(Yellow)")
##plt.savefig("NormalDE(RED),bDE31(Blue),NormalFE(Green),bFE31(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE31[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE31[:100],'y')
plt.title("NormalDE(RED),urDE31(Blue),NormalFE(Green),urFE31(Yellow)")
##plt.savefig("NormalDE(RED),urDE31(Blue),NormalFE(Green),urFE31(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE32[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE32[:100],'y')
plt.title("NormalDE(RED),IRDE32(Blue),NormalFE(Green),IRFE32(Yellow)")
##plt.savefig("NormalDE(RED),IRDE32(Blue),NormalFE(Green),IRFE32(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE32[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE32[:100],'y')
plt.title("NormalDE(RED),bDE32(Blue),NormalFE(Green),bFE32(Yellow)")
##plt.savefig("NormalDE(RED),bDE32(Blue),NormalFE(Green),bFE32(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE32[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE32[:100],'y')
plt.title("NormalDE(RED),urDE32(Blue),NormalFE(Green),urFE32(Yellow)")
##plt.savefig("NormalDE(RED),urDE32(Blue),NormalFE(Green),urFE32(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,irDE33[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,irFE33[:100],'y')
plt.title("NormalDE(RED),IRDE33(Blue),NormalFE(Green),IRFE33(Yellow)")
##plt.savefig("NormalDE(RED),IRDE33(Blue),NormalFE(Green),IRFE33(Yellow)")
##plt.show()


plt.plot(t,nMainDE[:100],'r')
plt.plot(t,bDE33[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,bFE33[:100],'y')
plt.title("NormalDE(RED),bDE33(Blue),NormalFE(Green),bFE33(Yellow)")
##plt.savefig("NormalDE(RED),bDE33(Blue),NormalFE(Green),bFE33(Yellow)")
##plt.show()

plt.plot(t,nMainDE[:100],'r')
plt.plot(t,urDE33[:100],'b')
plt.plot(t,nMainFE[:100],'g')
plt.plot(t,urFE33[:100],'y')
plt.title("NormalDE(RED),urDE33(Blue),NormalFE(Green),urFE33(Yellow)")
##plt.savefig("NormalDE(RED),urDE33(Blue),NormalFE(Green),urFE33(Yellow)")
##plt.show()


nMainDETrain = nMainDE[ :(len(nMainDE)//3) ]
nMainDETest = nMainDE[ (len(nMainDE)//3)+1 : ]
nMainFETrain = nMainFE[ : len(nMainFE)//3 ]
nMainFETest = nMainFE[ (len(nMainFE)//3)+1 : ]


irFE10Train = irFE10[ : (len(irFE10)//3)]
irFE10Test = irFE10[  (len(irFE10)//3)+1 :  ]

irDE10Train = irDE10[ : (len(irDE10)//3)]
irDE10Test = irDE10[  (len(irDE10)//3)+1 :  ]

##################### Train  IR DE ################################

Xtrain = []
for j in nMainDETrain:
    Xtrain.append(j)
for j in irDE10Train:
    Xtrain.append(j)
##Xtrain.append(irDE10Train)
print(np.shape(nMainDETrain))
print(np.shape(irDE10Train))
print(len(Xtrain))
Xtrain = np.asarray(Xtrain).reshape(-1,1)

Ytrain = []
for i in nMainDETrain:
    Ytrain.append(0)
for i in irDE10Train:
    Ytrain.append(1)
print(len(Ytrain))
##Ytrain = np.asarray(Ytrain).reshape(-1,1)

Xtest = []
for j in nMainDETest:
    Xtest.append(j)
for j in irDE10Test:
    Xtest.append(j)
Xtest = np.asarray(Xtest).reshape(-1,1)


Ytest =[]
for i in nMainDETest:
    Ytest.append(0)
for i in irDE10Test:
    Ytest.append(1)

print('''DE PREDICTION ONLY''')
rf = RandomForestRegressor()
rf.fit(Xtrain,Ytrain)
predictions = rf.predict(Xtest)
errors = abs(predictions - Ytest)
print('Mean Absolute Error:', round(np.mean(errors), 2))
accuracy = accuracy_score(predictions.round(),Ytest)
print("RandomForest DE only ", accuracy)

LR = LinearRegression().fit(Xtrain,Ytrain)
predictions=LR.predict(Xtest)
accuracy = accuracy_score(predictions.round(),Ytest)
print("LinearRegression DE only ", accuracy)

##################### Train  IR FE ################################

Xtrain = []
for j in nMainFETrain:
    Xtrain.append(j)
for j in irFE10Train:
    Xtrain.append(j)
##Xtrain.append(irDE10Train)
print(np.shape(nMainFETrain))
print(np.shape(irFE10Train))
print(len(Xtrain))
Xtrain = np.asarray(Xtrain).reshape(-1,1)

Ytrain = []
for i in nMainFETrain:
    Ytrain.append(0)
for i in irFE10Train:
    Ytrain.append(1)
print(len(Ytrain))
##Ytrain = np.asarray(Ytrain).reshape(-1,1)

Xtest = []
for j in nMainFETest:
    Xtest.append(j)
for j in irFE10Test:
    Xtest.append(j)
Xtest = np.asarray(Xtest).reshape(-1,1)


Ytest =[]
for i in nMainFETest:
    Ytest.append(0)
for i in irFE10Test:
    Ytest.append(1)

print('''FE PREDICTION ONLY''')

rf = RandomForestRegressor()
rf.fit(Xtrain,Ytrain)
predictions = rf.predict(Xtest)
errors = abs(predictions - Ytest)
print('Mean Absolute Error:', round(np.mean(errors), 2))
accuracy = accuracy_score(predictions.round(),Ytest)
print("RandomForest FE only ", accuracy)

LR = LinearRegression().fit(Xtrain,Ytrain)
predictions=LR.predict(Xtest)
accuracy = accuracy_score(predictions.round(),Ytest)
print("LinearRegression FE only ", accuracy)




