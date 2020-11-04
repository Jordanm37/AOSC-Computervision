# AOSC project

## Cross correlation in 1D and 2D
1. cd signalMatching
2. Run `python spatial_signal_offset.py` to excectute the 1D spatial cross correlation algortithm on files `sensorData1.txt` and `sensorData2.txt`. 
3. To toggle SSD, set `use_SSD = True` on line `xx`.
4. To toggle the library function, set `use_library = True` on line `xx`. 
5. To toggle the handmade cross correlation, set `use_handmade = True` on line `xx`. Currently this method takes approximately one hour. 
2. Run `python spatial_signal_offset.py` to excectute the 1D spatial cross correlaion algortithm on files `sensorData1` and `sensorData2`. 
3. To toggle SSD, set `use_SSD = True` on line `xx`.
4. To toggle the library function, set `use_library = True` on line `xx`. 
5. To toggle the handmade cross correlation, set `use_handmade = True` on line `xx`. Currently this method takes approximately one hour. 
6. Run `python singal_ffset_FT.py` to excecute the 1D spectral cross correlation algorithm on files `sensorData1` and `sensorData2`. 
7. cd imageMatching
8. Run `python where_wally.py` to excectute the 2D spatial cross correlation algortithm on files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. 
9. Run `python where_wally_FT.py` to excectute the 2D spectral cross correlation algortithm on files `wally_puzzle_rocketman.png` and `wallypuzzle_png.png`. 

## Depth Mapping
1. cd depthMapping
2. Run `depth_map.py` 

## Moth Eye image analysis
