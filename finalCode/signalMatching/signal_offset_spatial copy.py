import numpy as np
import time
import matplotlib.pyplot as plt
from signal_offset_spatial_functions import *

#import cProfile


'''
Program structure:

Read to singals.
Select from any combination of three methods:
A. SSD by hand
B: Library function
C: Handmade cross correlation
Plot out and 


'''

def main():
    #need to fix debug
    debug = False 
    use_SSD = False #Calculate using library functions and mean
    use_library = True 
    use_convolution = False
    #Read signal data and summarise
    s1_Data = read_file( "sensor1Data.txt" ) 
    s2_Data = read_file( "sensor2Data.txt" )
    print("Sensor-1 Data length = %d"%len(s1_Data))
    print("Sensor-2 Data length = %d"%len(s2_Data))
    visualise_signals(s1_Data, s2_Data)
    if debug:
        num_points=10000
        s1_Data, s1_Data = fewerDataPoints(s1_Data, s2_Data, num_points)
        
    '''Turn these into general functions eg:
    def correlation_find(s1_Data, s2_Data, method ):
    method(s1, s2)
        if used_dd: corelation_finD(, ,ssd)
        if jnsfvjvb: corelation_finD(, , use library)
    '''
    
    #This method uses the mean and standard deviation to remove noise from the signal data
    if use_SSD:       
        npts = len( s1_Data )
        lags = np.arange(-npts + 1, npts)        
        time_start = time.time()
        CCR, NormCCR = SSD_method( s1_Data, s2_Data )
        t_total = time.time() - time_start
        lags = np.arange(-npts + 1, npts)
        visualise_ccr(lags[0: npts],NormCCR)
        maxlag = lags[np.argmax(NormCCR)]
        print("\nSSD\n")
        print("\nCross Correlation = %.3f"%CCR)
        print("Normalized Cross Correlation = %.3f"%NormCCR[maxlag])
        print("\nmax correlation is at lag %d" % maxlag)
        print( "\nRun time = %.2f"%t_total )
        
    if use_library:
        npts = len( s1_Data )
        lags = np.arange(-npts + 1, npts)
        time_start = time.time()
        CCR, NormCCR = library_method( s1_Data, s2_Data )        
        t_total = time.time() - time_start
        visualise_ccr(lags,NormCCR)
        maxlag = lags[np.argmax(NormCCR)]        
        print("\nSSD with library\n")
        print("\nCross Correlation = %.3f"%CCR[maxlag])
        print("Normalized Cross Correlation = %.3f"%NormCCR[maxlag])
        print("\nmax correlation is at lag %d" % maxlag)
        print( "\nRun time = %.2f"%t_total )

    #Do without SSD
    if use_convolution:
        time_start = time.time()           
        offset, NormCCR = find_offset( s1_Data, s2_Data, debug )
        t_total = time.time() - time_start
        #Calculate domain of lagged times
        npts = len(s1_Data) - 1
        lags = np.arange(-npts , npts)
        visualise_ccr(lags,NormCCR)
        maxlag = lags[np.argmax(NormCCR)]
        print("\nCross Correlation with convolution\n")
        print("Normalized Cross Correlation = %.3f"%NormCCR)
        print("\nmax correlation is at lag %d" % maxlag)
        print( "\nRun time = %.2f"%t_total )
        
    #Singal separation summary
    offset = maxlag
    Freq = 44000
    sample_period = 1 / 44100
    speed_sound = 333 
    offset_time = offset * sample_period
    sensor_distance = abs(offset * sample_period * speed_sound)
    print("\nSignal separation summary")
    print("\nFreq. = %d"%Freq)  
    print("Off-Set = %d"%offset)
    print("Off-Set Time = %.3f"%offset_time)
    print("\nDistance between two sensors = %.2f meters"%sensor_distance)              
     
    
if __name__ == '__main__':
    
    main()


