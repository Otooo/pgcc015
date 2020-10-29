# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import Perceptron
from neurals import PerceptronMultiCamada
from neurals import Adaline
from neurals import Kohonen
from neurals import Fuzzy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


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
  # Set of testing
  inputTest = train.read_test('EPC01/teste.txt')

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


def startAdaline(n, e):
  train = Train()
  adaline = Adaline(n, e)

  # Set of training
  inputTrain, outputTrain = train.read_train('EPC02/dados.txt')
  # Set of testing
  inputTest = train.read_test('EPC02/teste.txt')

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
      elif typeAI == 'glass':
        fileNameTrain = f''
        fileNameTest  = f''
        fileNameValidation = ''

      # Create net
      perceptronMC = PerceptronMultiCamada(n, e, topology, momentum)

      # Set type dataset
      perceptronMC.set_type(typeAI)

      # Start execute time 
      start = time.time()

      # Set of training
      inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
      #normalizando entradas
      inputTrain = normalize(inputTrain).tolist()

      # Set of testing
      inputValidate, outputValidate = train.read_test2(fileNameTest, ',')
      # Set of validation
      # inputValidate, outputValidate = train.read_test2(fileNameValidation, ',')
      #normalizando
      inputValidate = normalize(inputValidate).tolist()
      
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

      epochKFold.append(epoch)
      eqmKFold.append(eqm[len(eqm)-1])
      timeKFold.append(timeExecuted)
      accuracyKFold.append(accuracy)

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


def startKohonen(n, r, topology, km=3):
  train = Train()

  kFold = 10
  fileNameTrain = f'EPC04/iris-10-fold/iris-10-{(kFold)}tra.dat'
  fileNameTest  = f'EPC04/iris-10-fold/iris-10-{(kFold)}tst.dat'

  # Create net
  kohonen = Kohonen(n, r, topology)

  # Start execute time 
  start = time.time()

  # Set of training
  inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
  # transformando em float as entradas
  inputTrain = np.array(inputTrain).astype(float).tolist()

  # Set of testing
  inputTest, outputTest = train.read_test2(fileNameTest, ',')
  # transformando em float as entradas
  inputTest = np.array(inputTest).astype(float).tolist()
  
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
  print('')
  # print('SAÍDA: ', output)

  plt_in   = plt.scatter(np.array(inputTest).T[0], np.array(inputTest).T[1], marker='^', c = 'green')
  plt_cent = plt.scatter(clusters.T[0], clusters.T[1], marker='x', s = 70, c = 'red')
  plt_n    = plt.scatter(np.array(neurons).T[0], np.array(neurons).T[1], marker='o', s = 60, c = 'blue')
  plt.legend((plt_in, plt_cent, plt_n),
           ('Entradas', 'Centroides', 'Neurônios'),
           scatterpoints=1,
           ncol=1,
           fontsize=8)
  # plt.grid() #função que desenha a grade no nosso gráfico
  plt.show()


def startFuzzy():
  ruleT = [[800, 900, 1000], [900, 1000, 1100], [1000, 1100, 1200]]
  ruleV = [[2.0, 4.5, 7.0],  [4.5, 7.0, 9.5],   [7.0, 9.5, 12.0]]
  ruleP = [[4, 5, 8],        [6, 8, 10],        [8, 11, 12]]

  # The last must be the result goal (Pressure)
  rules = {"Temperature":ruleT, "Volume":ruleV, "Pressure":ruleP}

  fuzzy = Fuzzy(rules)
  # fuzzy.relevances()

  fuzzy.set_rules()

  #a)
  outputNumber, simu = fuzzy.control(965, 11)
  print(f'SAÍDA a): {outputNumber} atm')
  fuzzy.relevances(simu)

  #b)
  outputNumber, simu = fuzzy.control(920, 7.6)
  print(f'SAÍDA b): {outputNumber} atm')
  fuzzy.relevances(simu)

  #c)
  outputNumber, simu = fuzzy.control(1050, 6.3)
  print(f'SAÍDA c): {outputNumber} atm')
  fuzzy.relevances(simu)

  #d)
  outputNumber, simu = fuzzy.control(843, 8.6)
  print(f'SAÍDA d): {outputNumber} atm')
  fuzzy.relevances(simu)

  #e)
  outputNumber, simu = fuzzy.control(1122, 5.2)
  print(f'SAÍDA e): {outputNumber} atm')
  fuzzy.relevances(simu)


#======================# GO GO GO #=====================#
# Perceptron EPC01
# print('PERCEPTRON: ')
# startPerceptron(0.01) # learning
# print('FIM PERCEPTRON')
# print()


# Adaline EPC02
# print('ADALINE: ')
# startAdaline(0.0025, 10**(-6)) # learning & precision
# print('FIM ADALINE')
# print()


# Perceptron Multicamada EPC03
# print('PERCEPTRON MULTICAMADA: ')
# print('Iris:')
# startPerceptronMultiCamada(0.1, 10**(-6), [4,2,3], 0, 'flower') # learning & precision & momentum & topology & momentum & type
# # --------------------------------------------------------------------------------------------------------------------------------
# print('')
# print('Wine')
# startPerceptronMultiCamada(0.1, 10**(-6), [4,4,4], 0, 'wine') # learning & precision & momentum & topology & momentum & type
# print('FIM PERCEPTRON MULTICAMADA')
# print()


# Kohonen EPC04
# print('KOHONEN: ')
# startKohonen(0.001, 1, 5) # learning & radius & topology
# print('FIM KOHONEN')
# print()


# Fuzzy EPC05
print('FUZZY: ')
startFuzzy()
print('FIM FUZZY')
print()

