#FastAnnoy
Term project for 2021 Fall "How to Write Fast Code"  
Baseline: Annoy (https://github.com/spotify/annoy)

## How to run
### Kernel Test
```asm
cd kernel_test
make all
```

### Installing baseline
```asm
pip3 install --user annoy==1.16
```

### Installing our version of implementation
```asm
python3 our_setup.py install --user
```
Then it will install the python library for our implementation

### Installing our version of implementation with openMP (not validated)
```asm
// first replace the content of annoylib.h with annoylib_omp.h
python3 omp_setup.py install --user
```
Then it will install the python library for our implementation

### Performance Measurement Codes
Indexing (Baseline v.s. Ours)
```asm
mkdir index
python3 indexing.py
```

Search (Baseline v.s. Ours)
```asm
python3 search.py
```
Output:
```asm
-bash-4.2$ python3 search.py
==================================================
Baseline implementation
Number of data: 10000 / Dimension: 256
[6768, 4385, 9582, 816, 7136, 4007, 960, 6462, 5839, 5446, 6992, 7425, 944, 5475, 6153, 1493, 2215, 356, 1930, 1038]
Average Query Time: 2.1013577779134114e-05


Number of data: 30000 / Dimension: 256
[6768, 12863, 21494, 23753, 18049, 10913, 27820, 14016, 10143, 13554, 9045, 16934, 15883, 8892, 27432, 4858, 27279, 10375, 7762, 6123]
Average Query Time: 2.540462811787923e-05


Number of data: 100000 / Dimension: 256
[96768, 95291, 88205, 94117, 18155, 64681, 95105, 1746, 71874, 49691, 84403, 50827, 97638, 31403, 28409, 93299, 79416, 97115, 95825, 39628]
Average Query Time: 4.319299062093099e-05


==================================================

Our implementation
Number of data: 10000 / Dimension: 256
[6768, 4385, 9582, 816, 7136, 4007, 960, 6462, 5839, 5446, 6992, 7425, 944, 5475, 6153, 1493, 2215, 356, 1930, 1038]
Average Query Time: 1.8262465794881184e-05


Number of data: 30000 / Dimension: 256
[6768, 12863, 21494, 23753, 18049, 10913, 27820, 14016, 10143, 13554, 9045, 16934, 15883, 8892, 27432, 4858, 27279, 10375, 7762, 6123]
Average Query Time: 2.0992469787597653e-05


Number of data: 100000 / Dimension: 256
[96768, 95291, 88205, 94117, 18155, 64681, 95105, 1746, 71874, 49691, 84403, 50827, 97638, 31403, 28409, 93299, 79416, 97115, 95825, 39628]
Average Query Time: 3.472544352213542e-05
```

## Contact
- Henry Oh (younghuo@andrew.cmu.edu)
- Fu Chi Ao (fuchia@andrew.cmu.edu)