# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import Kohonen
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling


def startKohonen(n, r, topology, km=3):
  train = Train()

  kFold = 7
  fileNameTrain = f'EPC04/iris-10-fold/iris-10-{(kFold)}tra.dat'
  fileNameTest  = f'EPC04/iris-10-fold/iris-10-{(kFold)}tst.dat'

  # Create net
  kohonen = Kohonen(n, r, topology)

  # Start execute time 
  start = time.time()

  # Set of training
  inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
  # normalizando entradas
  inputTrain = minmax_scaling(np.array(inputTrain), columns=[i for i in range(len(inputTrain[0][0]))]).tolist()

  # Set of testing
  inputTest, outputTest = train.read_test2(fileNameTest, ',')
  # normalizando entradas
  inputTest = minmax_scaling(np.array(inputTest), columns=[i for i in range(len(inputTest[0][0]))]).tolist()
  
  # Train 
  epoch, uMatrix, neurons = kohonen.kohonen_fit(inputTrain)

  # Validation
  output, clusters = kohonen.kohonen_predict(inputTest, km, outputTest)

  # End execute time
  end = time.time()
  timeExecuted = end - start

  #cmap='gray', # black => close
  plt.imshow(uMatrix, cmap='gray', interpolation='lanczos')
  plt.title(f'FOLD: {kFold}  Matriz {topology}x{topology}')
  plt.show()

  print(f'MATRIZ: {topology}x{topology}')
  print(f'FOLD: {kFold}')
  for ig in range(len(output)):
    print(f'  > Grupo {ig+1}: \n    {output[ig]} \n')
  print('ÉPOCAS: ', epoch)
  print(f'TEMPO: {(timeExecuted + 0.5)//60} minuto(s)')

  plt_in   = plt.scatter(np.array(inputTest).T[0], np.array(inputTest).T[1], marker='^', c = 'green')
  plt_cent = plt.scatter(clusters.T[0], clusters.T[1], marker='x', s = 70, c = 'red')
  plt_n    = plt.scatter(np.array(neurons).T[0], np.array(neurons).T[1], marker='o', s = 60, c = 'blue')
  plt.legend((plt_in, plt_cent, plt_n),
           ('Entradas', 'Centroides', 'Neurônios'),
           scatterpoints=1,
           ncol=1,
           fontsize=8)
  # plt.grid() #função que desenha a grade no gráfico
  plt.show()


#======================# GO GO GO #=====================#

# Kohonen EPC04
print('KOHONEN: ')
startKohonen(0.001, 1, 5) # learning & radius & topology
print('FIM KOHONEN')
print()
