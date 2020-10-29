# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from modules import Train
from neurals import FuzzyClassifier
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


def startFuzzyClassifier(step=1, withoutWeights):
  train = Train()

  eqmKFold = []
  accuracyKFold = []
  epochKFold = []
  timeKFold = []

  kFold = 1
  for i in range(kFold):
    fileNameTrain = f'EPC06/iris-10-fold/iris-10-{(i+1)}tra.dat'
    fileNameTest  = f'EPC06/iris-10-fold/iris-10-{(i+1)}tst.dat'

    # Set of training
    inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
    #normalizando entradas
    inputTrain = minmax_scaling(np.array(inputTrain), columns=[i for i in range(len(inputTrain[0][0]))]).tolist()

    # Set of testing
    inputTest, outputTest = train.read_test2(fileNameTest, ',')
    #normalizando
    inputTest = minmax_scaling(np.array(inputTest), columns=[i for i in range(len(inputTest[0][0]))]).tolist()

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

# Fuzzy EPC06
print('FUZZY: ')
startFuzzyClassifier(step=1) # passos para o epc
print('FIM FUZZY')
print()

