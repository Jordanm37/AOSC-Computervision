# Pattern matching and Computer Vision

Each folder represents an individual program and has been appropriately named to identify its application.
Note that the program for Moth Eye classification has been written in Python using OpenCV. 
Other programs have been written in Python and Numpy, Matplotlib, Scipy (without OpenCV or other CV libraries).

## Required Packages
- Python 3.7
- numpy
- scipy
- OpenCV
- matplotlib
- PIL

## Compilation and Execution Instructions

## Data
The python files read data from a hardcoded path to the local folder. 

## Cross correlation in 1D and 2D

To excecute 1D spatial cross correlation algortithm, follow these steps:
1. cd exp1_signalMatching
2. Run `python signal_offset_temporal.py` to process files `sensorData1.txt` and `sensorData2.txt` 
3. To toggle SSD, set `use_SSD = True` on line `13`.
4. To toggle the library function, set `use_library = True` on line `14`. 
5. To toggle the handmade cross correlation, set `use_convolution = True` on line `15`. Currently this method takes approximately one hour. 
6. To toggle the faster handmade convolution method, set `speed_up = True` on line `16`. .<br>

To excecute 1D spectral cross correlation algortithm, run `python signal_offset_FT.py` to process files `sensorData1` and `sensorData2`. 

To excecute 2D spatial cross correlation algortithm, follow these steps:<br>
1. cd imageMatching
2. Run `python spatial_2D_image_match.py` to process files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. <br>

To excecute 2D spectral cross correlation algortithm, run `python spectral_2D_image_match.py` to process files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. 

## Depth Mapping
To excecute the stereo vision depth mapping:
1. cd exp2_depthMapping
3. Run `dot_detection_CONV.py` and `dot_detection_cv.py` with calibration images to generate polynomial fit.  
3. Run `dot_calibration.py`
4. Run `depth_mapping_cv.py` and pass the local location of the desired images. 

## Moth Eye image analysis
1. cd extensionMoth
2. Run `circles.py`, toggle DIsplay to see statistics plots
