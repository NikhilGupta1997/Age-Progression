#!/bin/bash

g++ -Ofast -march=native -ggdb `pkg-config --cflags --libs opencv` $1 -o run 
