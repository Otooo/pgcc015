# -*- Coding: UTF-8 -*-
#coding: utf-8

import time
import math
import statistics
from functools import reduce
from neurals import Fuzzy
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import minmax_scaling


def startFuzzy():
  ruleT = [[800, 900, 1000], [900, 1000, 1100], [1000, 1100, 1200]]
  ruleV = [[2.0, 4.5, 7.0],  [4.5, 7.0, 9.5],   [7.0, 9.5, 12.0]]
  ruleP = [[4, 5, 8],        [6, 8, 10],        [8, 11, 12]]
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
# Fuzzy EPC05
print('FUZZY: ')
startFuzzy()
print('FIM FUZZY')
print()
