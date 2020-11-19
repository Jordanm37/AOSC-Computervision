import numpy as np
import time
import matplotlib.pyplot as plt
from signal_offset_temporal_functions import *

Freq = 44000
sample_period = 1 / 44100
speed_sound = 333 

def main():

    use_SSD = False #Calculate using library functions and mean
    use_library = False 
    use_convolution = True
    speed_up = True
    nth = False #to pick every nth element as a sample, set nth to number > 2
    
    #Read signal data and summarise
    s1_Data = read_file( "sensor1Data.txt" ) 
    s2_Data = read_file( "sensor2Data.txt" )
    
    print("Sensor-1 Data length = %d"%len(s1_Data))
    print("Sensor-2 Data length = %d"%len(s2_Data))
    
    visualise_signals(s1_Data, s2_Data)
    # if nth > 0:
    #     s1_Data, s1_Data = fewer_data_points(s1_Data, s2_Data, nth)
    npts = len( s1_Data )
         
    #This method uses the mean and standard deviation to remove noise from the signal data
    if use_SSD:       
        lags, time_start = init_vars(npts)
        ccr, NormCCR = ssd_method( s1_Data, s2_Data )
        t_total = time.time() - time_start
        maxlag = lags[np.argmax(NormCCR)]
        visualise_ccr(lags[0: npts],NormCCR, "SSD" )
        print_summary("SSD", NormCCR[maxlag], maxlag, t_total)
        print("\nCross Correlation = %.3f" % ccr)
        
    if use_library:
        lags, time_start = init_vars(npts)
        ccr, NormCCR = library_method( s1_Data, s2_Data )        
        t_total = time.time() - time_start
        maxlag = lags[np.argmax(NormCCR)]  
        visualise_ccr(lags,NormCCR,"SSD with library" )
        print_summary("SSD with library", NormCCR[maxlag], maxlag, t_total)
        print("\nCross Correlation = %.3f" % ccr[maxlag])

    # Do without SSD
    if use_convolution:
        npts -=1 
        lags, time_start = init_vars(npts, 0)      
        offset, NormCCR,  = find_offset( s1_Data, s2_Data, speed_up )
        t_total = time.time() - time_start
        maxlag = lags[-np.argmax(NormCCR)]
        #Calculate domain of lagged times
        visualise_ccr(lags,NormCCR[::-1], "convolution" )
        print_summary("Cross Correlation with convolution", NormCCR[maxlag], maxlag, t_total)
      
    # Signal separation summary TURN INTO FUNCTION
    offset = maxlag
    print_calculation_summary( offset, sample_period, speed_sound, Freq )
  
if __name__ == '__main__':
    
    main()
