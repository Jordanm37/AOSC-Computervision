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
The  data sets for expereiment 1 are in exp1_patternMatching, xxxxxxxxxxxxxxxxx
"" ""
THe python files read data from a hardcoded path. 

## Cross correlation in 1D and 2D

To excecute 1D spatial cross correlation algortithm, follow these steps:
1. cd exp1_signalMatching
2. Run `python spatial_signal_offset.py` to process files `sensorData1.txt` and `sensorData2.txt` 
3. To toggle SSD, set `use_SSD = True` on line `xx`.
4. To toggle the library function, set `use_library = True` on line `xx`. 
5. To toggle the handmade cross correlation, set `use_handmade = True` on line `xx`. Currently this method takes approximately one hour. 
6. To toggle the faster handmade convolution method, set `speed_up = True` on line cc. 

To excecute 1D spectral cross correlation algortithm, follow these steps:
1. Run `python signal_offset_FT.py` to process files `sensorData1` and `sensorData2`. 

To excecute 2D spatial cross correlation algortithm, follow these steps:
7. cd imageMatching
8. Run `python where_wally.py` to process files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. 

To excecute 2D spectral cross correlation algortithm, follow these steps:
9. Run `python where_wally_FT.py` to process files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. 

## Depth Mapping
To excecute the stereo vision depth mapping:
1. cd exp2_depthMapping
2. Run `depth_map.py` to process  

## Moth Eye image analysis
1. cd extensionMoth
2. Run `circles.py`