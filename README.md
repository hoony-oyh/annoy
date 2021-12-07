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

## Contact
- Henry Oh (younghuo@andrew.cmu.edu)
- Fu Chi Ao (fuchia@andrew.cmu.edu)