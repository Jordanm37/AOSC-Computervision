import numpy as np
import time
import math
import os
import matplotlib.pyplot as plt
from itertools import islice


def fewerDataPoints(data_1, data_2, nth):
#Debugging size for smaller runs 

    return data_1[::nth],data_2[::nth] 

#version 2 SSD
#Added functions to improve computation time 
#1
def CalcMean(lst):
    Sum = 0
    for i in range(0,len(lst)):
        Sum = Sum + lst[i]
    Mean = Sum/len(lst)    
    return(Mean)

#2
def CalcSD(lst,Mean):
    SD = 0
    for i in range(0,len(lst)):
        SD = SD + (lst[i]-Mean)*(lst[i]-Mean)
    SD = math.sqrt(SD/len(lst))        
    return(SD)    

#3
def CalcCrossCorr(f,g,fmean,gmean):
    Sum=0
    for i in range(0,len(f)):
        Sum = Sum + (f[i]-fmean)*(g[i]-gmean)
    CrossCorr = Sum/len(f)
    return(CrossCorr)

#4    
def CalcNormalisedCrossCorr(f,g,fmean,gmean,fsd,gsd):
    Sum = 0
    NormCrossCorr = np.zeros(len(f))
    for i in range(0,len(f)):
        Sum = Sum + (f[i]-fmean)*(g[i]-gmean)
        NormCrossCorr[i] = Sum/(len(f)*fsd*gsd)
    return NormCrossCorr

def SSD_method( s1_Data, s2_Data):

    S1_Mean = CalcMean( s1_Data )
    S2_Mean = CalcMean( s2_Data )

    S1_sdev = CalcSD( s1_Data, S1_Mean )
    S2_sdev = CalcSD( s2_Data, S2_Mean ) 

    CCR = CalcCrossCorr( s1_Data, s2_Data,S1_Mean, S2_Mean)
    NormCCR = CalcNormalisedCrossCorr( s1_Data, s2_Data, S1_Mean, S2_Mean, S1_sdev, S2_sdev)

    print("\nMean of Sensor Data-1 = %f"%S1_Mean)
    print("Mean of Sensor Data-2 = %f"%S2_Mean)

    print("\nStandard Deviation of Sensor Data-1  = %.3f"%S1_sdev)
    print("Standard Deviation of Sensor Data-2  = %.3f"%S2_sdev)

    return CCR, NormCCR


#library

def normalise(ccov, y1, y2, s1_Data):
    npts = len( s1_Data )
    return ccov / (npts * y1.std() * y2.std())

def library_method(s1_Data, s2_Data):
    #Use library functions to find CCR

    y1 = np.array( s1_Data )
    y2 = np.array( s2_Data )
    y1_mean_shift = y1 - y1.mean()         
    y2_mean_shift = y2 - y2.mean()
    ccov = np.correlate(y1_mean_shift, y2_mean_shift, mode='full')
    n_ccor = normalise(ccov, y1, y2, s1_Data)

    return ccov, n_ccor


#Handmade
def calculate_energy( p, g_slice ): 
    """
    Normalisation for 1D slice of N size array of same length of pattern passed. norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
    Inputs:
        p         list    Pattern must be non empty and sum-square of its elements precalculated and passed
		g_slice   list    slice of t from i to offset
    Output: 
        norm      Scalar  Float of variance for a given slice of the template/search and pattern
     """
    a = g_slice ** 2
    # b = a.sum()
    b = a.sum()
    c = p * b
    #norm = np.sqrt( p * ( g_slice**2).sum() )
    norm = np.sqrt(c) 
    return norm

#@jit(nopython=True)
def calculate_score( pattern, template):
    """
    Correlation for 1D slice of N size template/search array with pattern at given offset. Sum(f[i]*g[i+m])
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty and sum-square of its elements precalculated and passed

        template   Template with similar dimensionality to pattern

        offset  Offset position in the template/search array
        
    Output:
    ----------------
        score  Scalar float of correlation score between pattern and template slice
     """    


    # score = 0 
    # i = 0
    # lenp = len(pattern)
    # while i < lenp:
    # #for i in range(len( pattern )):
    #     p = pattern[i] 
    #     t = template[i + offset]
    #     i += 1
    #     if t > 0 and p > 0:
    #     	score += p * t
    # return score

# begin slice: max(p, offset)
# end slice: min(p + offset, p + t -1)

# pattern *template[max(p, offset):min(p + offset, p + t -1) ]

# p[p-1] * t[p-1]
# p[p-2:p-1] * t[p-1: p]]
# p[p-3:p-1] * t[p-1: p+1]
# p[p-3:p-1] * t[p:p+2]


    return (pattern * template).sum()
        
    # p = pattern.shape[0]
    # t = pattern.shape[0]

    # t_slice = template[offset : np.min(t, offset + p)]
    # p_slice = pattern[:t_slice.shape[0]]

    # return np.dot(t_slice, p_slice)



def zero_padding( pattern, template ):
    """
    Pad 1D template at begining and end of array with pattern length
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty and sum-square of its elements precalculated and passed

        template   Template with similar dimensionality to pattern
        
    Output:
    ----------------
        template_padded  Padded template array 
     """    
    
    #Calculate pad size 
    pad = [ 0 ] * ( len( pattern ) - 1 )
    #Pad begining and end of temple -1 for first element
    template_padded = pad + list(template) + pad

    return template_padded


#function that finds the largest element and its index in an array
def find_best_match( score ):
    """
    Find max value in 1D array and its index
 
    Inputs:
    ----------------
        score   1D target array
        
    Output:
    ----------------
        max_element Max Element in the array

        index   Index of largest element 

     """       
    s = np.array( score )
    try:
        max_element = np.amax( s )
    except:
        print( "Line 45 Error", score )
    index = np.argmax( s )

    return index, max_element


def norm_cross_corr( pattern, template, speed ): #change later to signal 1 and 2 as inputs
    """
    Normed cross correlation of two 1D arrays 
 
    Inputs:
    ----------------
        pattern   list[float]   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        norm_scores  Normed cross correlation array
     """       
    
    len_p = len(pattern)
    
    #Pad and initalise arrays for calculation   
    template_padded = zero_padding( pattern, template )
    corr_len = len( template_padded ) - len_p
    #scores = [0.0] * corr_len #must cast as float for later calculations, prevent error (numba requires all floats)
    #norm = [0.0] * corr_len 
    norm_scores = [0.0] * corr_len 
    #Precalculate pattern squared-sum and store, reduces calculation time by half 
    pattern_arr = np.array( pattern ) 
    import pdb; pdb.set_trace()
    pattern_sq_sum = ( pattern_arr ** 2 ).sum() #to use in norm - memoised values to reduce number of computations    
    template_pad_arr = np.array( template_padded )
    #Find normed cross correlation from convolution of pattern with template array slices
    len_t = len(template_pad_arr)
    
    t_start = time.time()
    #If speed is true, slices are taken without padding for the cross-corr
    for i in range( corr_len ):
        t_step = time.time()
        g_slice = template_pad_arr[ i : i + len_p ] 
        if speed:
            temp_slice_start = max(i, len_p -1 )
            temp_slice_end = min(len_t -len_p , i + len_p)
            pattern_slice_start = max(0, len_p -1 - i)
            pattern_slice_end = min(len_p, len_p - ( (i + len_p) - (len_t - len_p)  ))
            #Add short explanation 
            temp_slice = template_pad_arr[ temp_slice_start : temp_slice_end ] 
            pat_slice = pattern_arr[pattern_slice_start : pattern_slice_end] 
            score_i = calculate_score( pat_slice, temp_slice) 
        else:
            score_i = calculate_score( pattern_arr, g_slice)     
        
        #Whenever the norm is zero, the cross correlation is not calculated 
        if  score_i != 0 : 
            norm_i = calculate_energy( pattern_sq_sum, g_slice)
            norm_scores[i] = score_i / norm_i # division could be optimized?
        if i%100 == 0:
            tn = time.time() - t_step
            print("time=",tn)
            print(f' norm= {len(norm_scores)}' )
            print( f' i = { i } step time =  { tn - t_step} run time =  { tn - t_step}')
    
    return norm_scores


def find_offset( sig1, sig2, speed ): 
    """
    1D array offset index and value from  cross correlation 
 
    Inputs:
    ----------------
        correlation_arr   Calculated array of cross correlation coefficients 
        
    Output:
    ----------------
        (best_score, best_match)  Index of offset found from cross correlation
     """     
    correlation_arr = norm_cross_corr( sig1, sig2, speed )  

    idx, maxval = find_best_match( correlation_arr ) #clean this up
    #print( best_match )

    # subtract padding: - (len - 1)
    offset = idx - len( sig1 ) + 1
    return offset, correlation_arr 



def read_file( fileName ):
    """
    Read input data file and filters for numerical values 
 
    Inputs:
    ----------------
        fileName   File path 
 
    Output:
    ----------------
        data_list  List of read of only numerical data values
    
    References:
        super9super9 bronze badges, et al. 
        “Read File from Line 2 or Skip Header Row.” 
        Stack Overflow, 1 May 1960, stackoverflow.com/questions/4796764/read-file-from-line-2-or-skip-header-row.    
    """  
    # with open(fileName) as file: ??
    #     data = file.read()
    data = open( fileName ,"r") 
    data_list = [float(line.strip() ) for line in islice(data, 1, None)] 
    data.close()
         
    return data_list

def visualise_signals(s1_Data, s2_Data):
    # fig = plt.figure(figsize=(10, 4))
    # SubPlotRow=1
    # SubPlotCol=3
    # npts = len( s1_Data )
    # t = np.linspace(0, len(s1_Data ), npts)

    # plt.subplot(SubPlotRow,SubPlotCol,1)
    # plt.plot(t,s1_Data, color = 'red')
    # plt.title("Sensor-1 Data Plot")
    # plt.grid()

    # plt.subplot(SubPlotRow,SubPlotCol,2)
    # plt.plot(t,s2_Data, color = 'blue')
    # plt.title("Sensor-2 Data Plot")
    # plt.grid()

    # plt.subplot(SubPlotRow,SubPlotCol,3)
    # plt.plot(t,s1_Data, color = 'red')
    # plt.plot(t,s2_Data, color = 'blue')
    # plt.title("Sensor-1 and Sensor-2 Combined Data Plot")
    # plt.grid()
    # plt.show()

    npts = len( s1_Data )
    t = np.linspace(0, len(s1_Data ), npts)
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t,s1_Data, color = 'red')
    plt.xlabel("time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sensor-1 Data Plot")
    plt.grid()
    plot_save("Sensor-1 Data Plot")
 
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t,s2_Data, color = 'blue')
    plt.xlabel("time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sensor-2 Data Plot")
    plt.grid()
    plot_save("Sensor-2 Data Plot")
  
    fig = plt.figure(figsize=(10, 4))
    plt.plot(t,s1_Data, color = 'red')
    plt.plot(t,s2_Data, color = 'blue')
    plt.xlabel("time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sensor-1 and Sensor-2 Combined Data Plot")
    plt.grid()
    plot_save("Sensor-1 and Sensor-2 Combined Data Plot")

def visualise_ccr(lags,n_ccor, label):
    fig = plt.figure(figsize=(10, 4))
    scale = np.amax(n_ccor)
    plt.plot(lags, n_ccor)
    plt.ylim(-1.1*scale, scale*1.1)
    plt.axhline(y=0, color ='r')
    plt.ylabel('cross-correlation')
    plt.xlabel('lag of Sensor-1 relative to Sensor-2')
    plt.title(label)
    plt.grid()
    plot_save(label)
    plt.show()


def print_summary(title, NormCCR, maxlag, t_total):
    print(title)
    print("Normalized Cross Correlation = %.3f"%NormCCR)
    print("\nmax correlation is at lag %d" %maxlag)
    print( "\nRun time = %.2f"%t_total )

def init_vars(npts, add=1):
    lags = np.arange(-npts + add, npts)
    t_s = time.time()
    
    return lags, t_s

def plot_save(label):
    plt.tight_layout()
    path = os.path.join("..","figures","1D_signal_temporal",f'fig_{label}.png')
    plt.savefig(path,dpi = 250)
    