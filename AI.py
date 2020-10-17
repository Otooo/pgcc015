# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import Perceptron
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling


def startPerceptron(n):
  train = Train()
  perceptron = Perceptron(n)

  # Set of training
  inputTrain, outputTrain = train.read_train('EPC01/dados.txt')
  # transformando em float as entradas
  inputTrain = np.array(inputTrain).astype(float)
  # transformando em float as saidas
  outputTrain = np.array(outputTrain).astype(float)
  
  # Set of testing
  inputTest = train.read_test('EPC01/teste.txt')
  inputTest = np.array(inputTest).astype(float)

  # Train 
  smartNeuronium, epoch = perceptron.perceptron_fit(inputTrain, outputTrain)

  # Test
  output = perceptron.perceptron_predict(inputTest, smartNeuronium)

  # Accuracy
  # accuracy = accuracy(output, [1,2,3,4,5])

  # Logging
  print('PESOS FINAIS: ', smartNeuronium.w)
  print('ÉPOCAS: ', epoch)
  print('SAÍDA: ', output)



#======================# GO GO GO #=====================#

# Perceptron EPC01
print('PERCEPTRON: ')
startPerceptron(0.01) # learning
print('FIM PERCEPTRON')
print()
