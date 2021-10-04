#!/bin/bash

for LR in $(cat lr.txt); do
    python3 DISTmain.py 0.7 32 $LR 1e-8 3 42
done