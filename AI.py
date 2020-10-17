# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import Adaline
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling


def startAdaline(n, e):
  train = Train()
  adaline = Adaline(n, e)

  # Set of training
  inputTrain, outputTrain = train.read_train('EPC02/dados.txt')
  # transformando em float as entradas
  inputTrain = np.array(inputTrain).astype(float).tolist()
  # transformando em float as saidas
  outputTrain = np.array(outputTrain).astype(float).tolist()

  # Set of testing
  inputTest = train.read_test('EPC02/teste.txt')
  # transformando em float as entradas
  inputTest = np.array(inputTest).astype(float).tolist()

  # Train 
  smartNeuronium, epoch, eqm = adaline.adaline_fit(inputTrain, outputTrain)

  # Test
  output = adaline.adaline_predict(inputTest, smartNeuronium)

  # Accuracy
  # accuracy = accuracy(output, [1,2,3,4,5])

  # Logging
  print('PESOS FINAIS: ', smartNeuronium.w)
  print('ÉPOCAS: ', epoch)
  print('SAÍDA: ', output)
  # plot
  plt.plot(range(epoch), eqm)
  plt.ylabel('EQM')
  plt.xlabel('Épocas')
  plt.show()


#======================# GO GO GO #=====================#

# Adaline EPC02
print('ADALINE: ')
startAdaline(0.0025, 10**(-6)) # learning & precision
print('FIM ADALINE')
print()
