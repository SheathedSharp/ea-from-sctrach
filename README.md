# Evolutionary Algorithm Optimizer

A simple evolutionary algorithm implementation for function optimization.

[Colab Notebook](https://colab.research.google.com/drive/1XxIlbgBI3WmErFJwRQDiuBdSByjIPWD1?usp=sharing)

## Overview
This project implements an evolutionary algorithm to optimize two different multi-modal functions:

### Function 1: Modified Sine Function
$f(x_1,x_2) = 3-\sin^2(jx_1)-\sin^2(jx_2)$

where:
- $x_1,x_2 \in [0,6]$
- For $j=2,3,4,5$, the function has 16, 36, 64, and 100 global optima respectively
- Current implementation uses $j=2$

### Function 2: Shubert Function
$f(x_1,...,x_n) = \prod_{i=1}^{n}\sum_{j=1}^{5}j\cos[(j+1)x_i+j]$

where:
- $x_i \in [-10,10]$
- For $n=2$, the function has 18 different global optima

## Features
- Population initialization
- Custom selection mechanism
- Crossover operation (rate: 0.4)
- Mutation operation (rate: 0.1)
- Real-time 3D visualization

## Requirements
- python
- numpy==1.21.0
- pandas==2.0.0
- matplotlib==3.5.0
