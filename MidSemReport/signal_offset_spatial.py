import numpy as np
import time
import matplotlib.pyplot as plt
from itertools import islice



def calculate_energy( p, t, offset, len_p ): # - memoise 
    """
    Normalisation for 1D slice of N size array of same length of pattern passed.
    norm= sqrt( sum(f[i]^2) * sum(g[m]^2) )
 
    Inputs:
    ----------------
        p   Pattern must be non empty and sum-square of its elements precalculated and passed

        t   Template with similar dimensionality to pattern

        offset  Offset position in the template/search array

        len_p   offset for end-of-slice index for the slice of template
        
    Output:
    ----------------
        norm  Scalar float of variance for a given slice of the template/search and pattern
     """
    g_slice = t[ offset : offset + len_p ] 
    norm = np.sqrt( p * ( g_slice**2).sum() ) 
    # if norm == 0 :
    #     print ("p=", p, "template=", g_slice, "offset = ", offset, "\n")
    return norm


def calculate_score( pattern, template, offset):
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
    score  = 0 
    #Mutltiply and add each element of the pattern and template
    for i in range(len( pattern )):
        o = i + offset
        #try:
        score += pattern[ i ] * template[ o ]
        #except:
        #    print( "Error line 26", pattern, template )

    return score



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

    return max_element, index


def n_corr( pattern, template): #change later to signal 1 and 2 as inputs
    """
    Normed cross correlation of two 1D arrays
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        norm_scores  Normed cross correlation array
     """       

    #Pad and initalise arrays for calculation   
    template = zero_padding( pattern, template )
    corr_len = len( template ) - len( pattern )
    scores = [0] * ( corr_len )
    norm = [0] * ( corr_len ) 
    norm_scores = [0] * ( corr_len ) 
    #test = [0] * ( len( template ) - len( pattern ) )
    #t_start = time.time()
    
    #Precalculate pattern squared-sum and store, reduces calculation time by half 
    pattern_arr = np.array( pattern )
    pattern_sq_sum = ( pattern_arr**2 ).sum() #to use in norm - memoised values to reduce number of computations
    template_arr = np.array( template )
    
    #Find normed cross correlation from convolution of pattern with template array slices
    for i in range( len( scores ) ):
        scores[ i ] = calculate_score( pattern, template, i)
        #print( scores )
        #Whenever the cross correlation is zero, the cross correlation is not calculated 
        if  scores[i]!=0 : 
            norm[ i ] = calculate_energy( pattern_sq_sum, template_arr, i, len(pattern))
            norm_scores[i] = scores[ i ]/norm[ i ]
        # tn = time.time()
        # print( f'{ i } run time =  { tn - t_start}')
        
        #print( "s=", scores,"\n", "n=", norm, "\n")

    return norm_scores

def find_offset(pattern, template): 
    """
    1D array offset index and value from normed cross correlation 
 
    Inputs:
    ----------------
        pattern   Pattern must be non empty 

        template   Template, search space with similar dimensionality to pattern
        
    Output:
    ----------------
        (best_score, best_match)  Index of offset found from cross correlation
     """     

    norm_corr = n_corr( pattern, template)

    best_score, best_match = find_best_match( norm_corr )
    #print( best_match )

    #Plot array of cross correlation
    plt.figure()
    plt.plot(norm_corr)

    # subtract padding: - (len - 1)
    return best_match - len( pattern ) + 1, best_score 



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
    
    data = open( fileName ,"r") 
    data_list = [float(line.strip() ) for line in islice(data, 1, None)] 
    data.close()
         
    return data_list



#calculate signal offset for files in the local directory that are read into program

def main():

    time_start = time.time()

    data_1 = read_file( "sensor1Data.txt" ) 
    data_2 = read_file( "sensor2Data.txt" )
    data_1_len = len(data_1)
    # print( data_1_len, len( data_2 ) )
    sample_period = 1 / 44100
    speed_sound = 333
    #Debugging size 
    size = data_1_len 

    offset, corr_value = find_offset( data_1[:size], data_2[:size] )

    offset_time = offset*sample_period

    sensor_distance = offset * sample_period * speed_sound

    t_total = time.time() - time_start
                
    print("offset time = ", offset_time, "offset position =", offset,"sensor distance =", sensor_distance, "run time = ", t_total )
    
    #plotting
    plt.figure()
    plt.subplot(211)  
    plt.plot(data_1[:size])
    plt.subplot(212)  
    plt.plot(data_2[:size])
    plt.show()   

 
   
    
if __name__ == '__main__':
    
    main()

