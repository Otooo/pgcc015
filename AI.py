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


def startFuzzyClassifier():
  train = Train()

  accuracyKFold = {
      'mrfc_wtp': [],
      'mrfg_wtp': [],
      'mrfc_wp':  [],
      'mrfg_wp':  []
  }
  foldRules = {}
  foldRulesWithPeso = {}

  kFold = 10
  for i in range(kFold):
    fileNameTrain = f'EPC06/iris-10-fold/iris-10-{(i+1)}tra.dat'
    fileNameTest  = f'EPC06/iris-10-fold/iris-10-{(i+1)}tst.dat'

    # Set of training
    inputTrain, outputTrain = train.read_train(fileNameTrain, ',')
    
    # Set of testing
    inputTest, outputTest = train.read_test2(fileNameTest, ',')

    # Without pesos
    rules = FuzzyClassifier.wang_mendel(inputTrain, outputTrain)
    foldRules[i+1] = rules
    print(f'Fold {i+1} com {len(rules)} Regras')
    # ...
    fuzzy = None
    fuzzy = FuzzyClassifier(rules)
    output_mrfc = fuzzy.classifier_MRFC(inputTest)
    output_mrfg = fuzzy.classifier_MRFG(inputTest)

    # With pesos
    pesosCF = FuzzyClassifier.calc_peso(inputTrain, outputTrain)
    mp = statistics.mean(pesosCF)
    print(f'Média dos pesos: {mp}')
    rules = FuzzyClassifier.wang_mendel(inputTrain, outputTrain, pesosCF)
    foldRulesWithPeso[i+1] = rules
    print(f'Fold {i+1} com {len(rules)} Regras Utilizando Pesos')
    # ...
    fuzzy = None
    fuzzy = FuzzyClassifier(rules)
    output_mrfcwp = fuzzy.classifier_MRFC(inputTest)
    output_mrfgwp = fuzzy.classifier_MRFG(inputTest)
    

    # Analysis
    accMrfc = calc_accuracy(output_mrfc, outputTest)
    accMrfg = calc_accuracy(output_mrfg, outputTest)
    accMrfcwp = calc_accuracy(output_mrfcwp, outputTest)
    accMrfgwp = calc_accuracy(output_mrfgwp, outputTest)
    accuracyKFold['mrfc_wtp'].append(accMrfc)
    accuracyKFold['mrfg_wtp'].append(accMrfg)
    accuracyKFold['mrfc_wp'].append(accMrfcwp)
    accuracyKFold['mrfg_wp'].append(accMrfgwp)

    print(f'Acurácia MRFC Fold {i+1}: {accMrfc}')
    print(f'Acurácia MRFC Fold com Peso {i+1}: {accMrfcwp}')
    print(f'Acurácia MRFG Fold {i+1}: {accMrfg}')
    print(f'Acurácia MRFG Fold com Peso {i+1}: {accMrfgwp}')
    print('')

  print('')
  # for i in range(len(foldRulesWithPeso[1])):
  #   print(foldRulesWith[1][i]['condition']) # Fold 1
  #   print(foldRulesWithPeso[1][i]['condition']) # Fold 1
  # print('')

  mfc = statistics.mean(accuracyKFold['mrfc_wtp'])
  dpfc = statistics.stdev(accuracyKFold['mrfc_wtp'])

  mfg = statistics.mean(accuracyKFold['mrfg_wtp'])
  dpfg = statistics.stdev(accuracyKFold['mrfg_wtp'])

  mfcwp = statistics.mean(accuracyKFold['mrfc_wp'])
  dpfcwp = statistics.stdev(accuracyKFold['mrfc_wp'])
  
  mfgwp = statistics.mean(accuracyKFold['mrfg_wp'])
  dpfgwp = statistics.stdev(accuracyKFold['mrfg_wp'])

  print(f'Média MRFC: {mfc}')
  print(f'Desvio Padrão MRFC: {dpfc}')  
  print('')
  print(f'Média MRFC com Peso: {mfcwp}')
  print(f'Desvio Padrão MRFC com Peso: {dpfcwp}')  
  print('')
  print(f'Média MRFG: {mfg}')
  print(f'Desvio Padrão MRFG: {dpfg}')  
  print('')
  print(f'Média MRFG com Peso: {mfgwp}')
  print(f'Desvio Padrão MRFG com Peso: {dpfgwp}')  


#======================# GO GO GO #=====================#

# Fuzzy EPC06
print('FUZZY: ')
startFuzzyClassifier()
print('FIM FUZZY')
print()
