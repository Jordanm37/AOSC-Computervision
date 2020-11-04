import numpy as np
import time
import matplotlib.pyplot as plt
import time
from signal_offset_FT_functions import *

# Use bandpass filter to remove singal noise, and show improvement on spatial cross corellation?
# function to read lines from file to list
     
def main():

    data_1 = np.array(read_file( "sensor1Data.txt")) 
    data_2 = np.array(read_file( "sensor2Data.txt"))
    
    #Constants
    Fs = 44100 #Sampling frequency
    sample_period = 1 / Fs 
    speed_m_sec = 333
    #Find offset
    len_d_1 = len(data_1)
    len_d_2 = len(data_2)
    npts = len_d_1 - 1
    lags = np.arange(-npts , npts)
    offset, corr_sig_1_2, t_total = find_best_lag(data_1, data_2,lags)
    offset_sec = offset * sample_period
    sensor_distance = abs( offset_sec * speed_m_sec )
    print("\nFreq. = %d"%Fs)
    print("\nMax correlation is at lag %d" %offset)
    print("Off-Set = %d"%offset)
    print("Off-Set Time = %.3f"%offset_sec)
    print("\nDistance between two sensors = %.2f meters"%sensor_distance)              
    print( "\nRun time = %.2f"%t_total )
    # print(fft_1, '\n')
    # print(corr, '\n')
    # print(offset, '\n')
    #remove mirrored frequency data
    fft_half_1 = remove_repeated(data_1,len_d_1, Fs, "Signal_1")
    fft_half_2 = remove_repeated(data_2,len_d_2, Fs, "Signal_2")
    # LPF
    low_pass(data_1, Fs, len_d_1, "Signal_1_lpf", fft_half_1 )
    low_pass(data_2, Fs, len_d_2, "Signal_2_lpf", fft_half_2 )
    #plotting
    plt.figure()
    plt.subplot(311)  
    plt.plot(data_1)
    plt.title("Signal_1_raw")
    plt.subplot(312)  
    plt.plot(data_2)
    plt.title("Signal_2_raw")
    plt.subplot(313)  
    plt.plot(lags[0:npts+1], corr_sig_1_2)
    plt.ylabel('cross-correlation')
    plt.xlabel('lag of Sensor-1 relative to Sensor-2')
    plt.show()   

if __name__ == '__main__':
    
    main()