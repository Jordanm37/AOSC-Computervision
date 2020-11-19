import numpy as np
import time
from signal_offset_FT_functions import *

#Constants
Fs = 44100 #Sampling frequency
sample_period = 1 / Fs 
speed_m_sec = 333

def main():

    data_1 = np.array(read_file( "sensor1Data.txt")) 
    data_2 = np.array(read_file( "sensor2Data.txt"))
        
	# Find offset
    len_d_1 = len(data_1)
    len_d_2 = len(data_2)
    npts = len_d_1 - 1
    lags = np.arange(-npts , npts)

    # remove mirrored frequency data
    fft_half_1 = remove_repeated(data_1,len_d_1, Fs, "Signal_1_half")
    fft_half_2 = remove_repeated(data_2,len_d_2, Fs, "Signal_2_half")
    
    # LPF
    low_pass(data_1, Fs, len_d_1, "Signal_1_lpf", fft_half_1, True )
    low_pass(data_2, Fs, len_d_2, "Signal_2_lpf", fft_half_2, True )

    # calculate sensor distance
    offset, corr_sig_1_2, t_total = find_best_lag(data_1, data_2,lags)
    offset_sec = offset * sample_period
    sensor_distance = abs( offset_sec * speed_m_sec )
    print_summary(Fs, offset, offset_sec, sensor_distance, t_total)
    visualise_ccr(lags[0:npts+1], corr_sig_1_2, "FT_ccr") 

if __name__ == '__main__':   
    main()