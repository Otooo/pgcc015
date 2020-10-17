import csv
import random
import threading
import logging


class Train:

  def __init__(self): 
    self.dataset = []


  def read_train(self, filename, delimiter=' ', lengthOutput=-1):
    with open(filename) as _file:
      data = csv.reader(_file, delimiter=delimiter)
      dataTrain = []
      for line in data:
        #any('@' in c for c in line):
        if '@' not in line[0]:
          line = [elemento.strip() for elemento in line]
          dataTrain.append(line)
      
      inputsWithBias, outputs = [], []
      for data in dataTrain:
        inputsWithBias.append(data[:-1])
        outputs.append(data[-1:][0])
      return inputsWithBias, outputs


  def read_test2(self, filename, delimiter=' ', lengthOutput=-1):
    with open(filename) as _file:
      data = csv.reader(_file, delimiter=delimiter)
      dataTest = []
      for line in data:
        #any('@' in c for c in line):
        if '@' not in line[0]:
          line = [elemento.strip() for elemento in line]
          dataTest.append(line)
      
      inputsTest, outputs = [], []
      for data in dataTest:
        inputsTest.append(data[:-1])
        outputs.append(data[-1:][0])
      return inputsTest, outputs


  def read_test(self, filename, delimiter=' '):
     with open(filename) as _file:
      data = csv.reader(_file, delimiter=delimiter)
      dataTest = []
      for line in data:
        if '@' not in line[0]:
          line = [elemento.strip() for elemento in line]
          dataTest.append(line)
      return dataTest
