# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import Perceptron
from neurals import Adaline
from neurals import PerceptronMultiCamada
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling


def calc_accuracy(output, outputValid):
  total = 0
  for i in range(len(output)):
    if (output[i] == outputValid[i]):
      total += 1
    else:
      pass
  return (total / len(outputValid)) * 100
  

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


def startPerceptronMultiCamada(n, e, topology, momentum, typeAI):
  train = Train()

  eqmKFold = []
  accuracyKFold = []
  epochKFold = []
  timeKFold = []

  # Exercise 1)
  # xTimes = 3
  xTimes = 1
  for xvez in range(xTimes):

    # 10-fold
    kFold = 10
    for i in range(kFold):
      fileNameTrain = ''
      fileNameTest  = ''
      if typeAI == 'flower':
        fileNameTrain = f'EPC03/iris-10-fold/iris-10-{(i+1)}tra.dat'
        fileNameTest  = f'EPC03/iris-10-fold/iris-10-{(i+1)}tst.dat'
        fileNameValidation = f'EPC03/iris-10-fold/iris-10-{(i+1)}val.dat'
      elif typeAI == 'wine':
        fileNameTrain = f'EPC03/winequality-white-10-fold/winequality-white-10-{(i+1)}tra.dat'
        fileNameTest  = f'EPC03/winequality-white-10-fold/winequality-white-10-{(i+1)}tst.dat'
        fileNameValidation = f'EPC03/winequality-white-10-fold/winequality-white-10-{(i+1)}tst.dat'

      # Create net
      perceptronMC = PerceptronMultiCamada(n, e, topology, momentum)

      # Set type dataset
      perceptronMC.set_type(typeAI)

      # Start execute time 
      start = time.time()

      # Set of training
      inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
      #normalizando entradas
      inputTrain = minmax_scaling(np.array(inputTrain), columns=[i for i in range(len(inputTrain[0][0]))]).tolist()

      # Set of testing
      inputValidate, outputValidate = train.read_test2(fileNameTest, ',')
      # Set of validation
      # inputValidate, outputValidate = train.read_test2(fileNameValidation, ',')
      #normalizando
      inputValidate = minmax_scaling(np.array(inputValidate), columns=[i for i in range(len(inputValidate[0][0]))]).tolist()
      
      # Train 
      epoch, eqm = perceptronMC.perceptro_m_c_fit(inputTrain, outputTrain)

      # Test / Validation
      # output = perceptronMC.perceptron_m_c_predict(inputTest, outputTest)
      output = perceptronMC.perceptron_m_c_predict(inputValidate, outputValidate)

      # End execute time
      end = time.time()
      timeExecuted = end - start

      # Accuracy
      accuracy = calc_accuracy(output, outputValidate)

      # Plot
      # if typeAI == 'flower':
      #   typeTitle = 'Iris Plants'
      # elif typeAI == 'wine':
      #   typeTitle = 'White Wine Quality'
      # elif typeAI == 'glass':
      #   typeTitle = 'Glass Identification'
      # plt.plot(range(1, epoch+1), eqm)
      # plt.ylabel('EQM')
      # plt.xlabel('Épocas')
      # plt.title(f'Tempo de execução {typeTitle}: {timeExecuted:.0f} segundos')
      # plt.grid(True)
      # plt.show()

      # epochKFold.append(epoch)
      # eqmKFold.append(eqm[len(eqm)-1])
      # timeKFold.append(timeExecuted)
      # accuracyKFold.append(accuracy)

  # Logging
  print('TOPOLOGIA: ', topology)
  print('MÉDIA ÉPOCAS: ', statistics.mean(epochKFold))
  print('MÉDIA EQM FINAL: ', statistics.mean(eqmKFold))
  print('MÉDIA TEMPO TOTAL (s): ', statistics.mean(timeKFold))
  print('MÉDIA ACURÁCIA: ', statistics.mean(accuracyKFold))

  print('DESVIO PADRÃO ÉPOCAS: ', statistics.stdev(epochKFold))
  print('DESVIO PADRÃO EQM FINAL: ', statistics.stdev(eqmKFold))
  print('DESVIO PADRÃO TEMPO TOTAL (s): ', statistics.stdev(timeKFold))
  print('DESVIO PADRÃO ACURÁCIA: ', statistics.stdev(accuracyKFold))


#======================# GO GO GO #=====================#

# Perceptron EPC01
print('PERCEPTRON: ')
startPerceptron(0.01) # learning
print('FIM PERCEPTRON')


# Adaline EPC02
print('ADALINE: ')
startAdaline(0.0025, 10**(-6)) # learning & precision
print('FIM ADALINE')
print()


# Perceptron Multicamada EPC03
print('PERCEPTRON MULTICAMADA: ')
print('Iris:')
startPerceptronMultiCamada(0.1, 10**(-6), [4,2,3], 0, 'flower') # learning & precision & momentum & topology & momentum & type
# --------------------------------------------------------------------------------------------------------------------------------
print('')
print('Wine')
startPerceptronMultiCamada(0.1, 10**(-6), [4,4,4], 0, 'wine') # learning & precision & momentum & topology & momentum & type
print('FIM PERCEPTRON MULTICAMADA')
print()

