from scipy.io import loadmat
import numpy as np


mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Normal\\97")
print(mat.keys())
nDE = []
nDE.append(mat['X097_DE_time'])
print(len(nDE))
nMainDE = []
for j in nDE:
    for k in j:
        nMainDE.append(k)
print(len(nMainDE))

nFE = []
nFE.append(mat['X097_FE_time'])
print(len(nFE))
nMainFE = []
for j in nFE:
    for k in j:
        nMainFE.append(k)
print(len(nMainFE))

##########################  Grade1 LOAD 0  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\109")
print(mat.keys())
irDE10 = []
irDE10 = list(mat['X109_DE_time'])
print(len(irDE10))

irFE10 = []
irFE10 = list(mat['X109_FE_time'])
print(len(irFE10))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\122")
print(mat.keys())
bDE10 = []
bDE10 = list(mat['X122_DE_time'])
print(len(bDE10))

bFE10 = []
bFE10 = list(mat['X122_FE_time'])
print(len(bFE10))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load0\\135")
print(mat.keys())
urDE10 = []
urDE10 = list(mat['X135_DE_time'])
print(len(urDE10))

urFE10 = []
urFE10 = list(mat['X135_FE_time'])
print(len(urFE10))

##########################  Grade1 LOAD 1  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\110")
print(mat.keys())
irDE11 = []
irDE11 = list(mat['X110_DE_time'])
print(len(irDE11))

irFE11 = []
irFE11 = list(mat['X110_FE_time'])
print(len(irFE11))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\123")
print(mat.keys())
bDE11 = []
bDE11 = list(mat['X123_DE_time'])
print(len(bDE11))

bFE11 = []
bFE11 = list(mat['X123_FE_time'])
print(len(bFE11))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load1\\136")
print(mat.keys())
urDE11 = []
urDE11 = list(mat['X136_DE_time'])
print(len(urDE11))

urFE11 = []
urFE11 = list(mat['X136_FE_time'])
print(len(urFE11))


##########################  Grade1 LOAD 2  #####################
 

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\111")
print(mat.keys())
irDE12 = []
irDE12 = list(mat['X111_DE_time'])
print(len(irDE12))

irFE12 = []
irFE12 = list(mat['X111_FE_time'])
print(len(irFE12))


############################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\124")
print(mat.keys())
bDE12 = []
bDE12 = list(mat['X124_DE_time'])
print(len(bDE12))

bFE12 = []
bFE12 = list(mat['X124_FE_time'])
print(len(bFE12))

#################################

mat = loadmat("C:\\Users\\sahilvaid\\Desktop\\Data\\Faulty\\48k\\grade1\\load2\\137")
print(mat.keys())
urDE12 = []
urDE12 = list(mat['X137_DE_time'])
print(len(urDE12))

urFE12 = []
urFE12 = list(mat['X137_FE_time'])
print(len(urFE12))



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
print(len(urFE10))

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














