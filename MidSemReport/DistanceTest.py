import os
import math
import numpy as np
import matplotlib.pyplot as plt


def CalcMean(lst):
    Sum = 0
    for i in range(0,len(lst)):
        Sum = Sum + lst[i]
    Mean = Sum/len(lst)    
    return(Mean)
   
def CalcSD(lst,Mean):
    SD = 0
    for i in range(0,len(lst)):
        SD = SD + (lst[i]-Mean)*(lst[i]-Mean)
    SD = math.sqrt(SD/len(lst))        
    return(SD)

    
def CalcCrossCorr(f,g,fmean,gmean):
    Sum=0
    for i in range(0,len(f)):
        Sum = Sum + (f[i]-fmean)*(g[i]-gmean)
    CrossCorr = Sum/len(f)
    return(CrossCorr)
    
def CalcNormalisedCrossCorr(f,g,fmean,gmean,fsd,gsd):
    Sum = 0
    for i in range(0,len(f)):
        Sum = Sum + (f[i]-fmean)*(g[i]-gmean)
    NormCrossCorr = Sum/(len(f)*fsd*gsd)
    return(NormCrossCorr)


ProjectPath = os.getcwd()
Sensor_1 = ProjectPath + "\\sensor1Data.txt"
Sensor_2 = ProjectPath + "\\sensor2Data.txt"

fpt1 = open(Sensor_1,"r")
fpt2 = open(Sensor_2,"r")

Sensor_1_Data = (fpt1.read()).split("\n")
Sensor_2_Data = (fpt2.read()).split("\n")
fpt1.close()
fpt2.close()

SubPlotRow=2
SubPlotCol=2

print("Sensor-1 Data length = %d"%len(Sensor_1_Data))
print("Sensor-2 Data length = %d"%len(Sensor_2_Data))

S1Data=[]
S2Data=[]
Count=100000
t = []
for c in range(0,Count):
    t.append(c+1)
    S1Data.append(float(Sensor_1_Data[c]))
    S2Data.append(float(Sensor_2_Data[c]))


MeanS1 = CalcMean(S1Data)
MeanS2 = CalcMean(S2Data)

SD1 = CalcSD(S1Data,MeanS1)
SD2 = CalcSD(S2Data,MeanS2)


CCR = CalcCrossCorr(S1Data,S2Data,MeanS1,MeanS2)
NormCCR = CalcNormalisedCrossCorr(S1Data,S2Data,MeanS1,MeanS2,SD1,SD2)


print("\nMean of Sensor Data-1 = %f"%MeanS1)
print("Mean of Sensor Data-2 = %f"%MeanS2)

print("\nStandard Deviation of Sensor Data-1  = %.3f"%SD1)
print("Standard Deviation of Sensor Data-2  = %.3f"%SD2)


print("\nCross Correlation = %.3f"%CCR)
print("Normalized Cross Correlation = %.3f"%NormCCR)

plt.subplot(SubPlotRow,SubPlotCol,1)
plt.plot(t,S1Data, color = 'red')
plt.title("Sensor-1 Data Plot")
plt.grid()


plt.subplot(SubPlotRow,SubPlotCol,2)
plt.plot(t,S2Data, color = 'blue')
plt.title("Sensor-2 Data Plot")
plt.grid()


plt.subplot(SubPlotRow,SubPlotCol,3)
plt.plot(t,S1Data, color = 'red')
plt.plot(t,S2Data, color = 'blue')
plt.title("Sensor-1 and Sensor-2 Combined Data Plot")
plt.grid()

npts = len(S1Data)
t = np.linspace(0, len(S1Data), npts)
y1 = np.array(S1Data)
y2 = np.array(S2Data)

lags = np.arange(-npts + 1, npts)
ccov = np.correlate(y1 - y1.mean(), y2 - y2.mean(), mode='full')
ccor = ccov / (npts * y1.std() * y2.std())


plt.subplot(SubPlotRow,SubPlotCol,4)
plt.plot(lags, ccor)
plt.ylim(-1.1, 1.1)
plt.ylabel('cross-correlation')
plt.xlabel('lag of Sensor-1 relative to Sensor-2')
plt.grid()

maxlag = lags[np.argmax(ccor)]
print("\nmax correlation is at lag %d" % maxlag)

Freq = 44000
offset = maxlag
sample_period = 1/Freq
speed_sound = 333

offset_time = offset*sample_period
sensor_distance = abs(offset * sample_period * speed_sound)

print("\nFreq. = %d"%Freq)
print("Off-Set = %d"%offset)
print("Off-Set Time = %.3f"%offset_time)
print("\nDistance between two sensors = %.2f meters"%sensor_distance)

plt.show()
