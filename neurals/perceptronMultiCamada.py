import os
import threading
import logging
from modules import Neuronium
from modules import Layer
import math

class PerceptronMultiCamada:

  def __init__(self, nLearning, precision, neuroniumsLayer=[], momentum=0, maxEpoch=5000): 
    self.nLearning = nLearning
    self.maxEpoch = maxEpoch
    self.precision = precision
    self.momentum = momentum

    # wine, flower or glass
    self.typeAI = None

    self.beta = 0.5

    self.layer = Layer()
    self.layer.label = 'first'
    self.neuroniumsLayer = neuroniumsLayer
    quantityLayers = len(neuroniumsLayer)
    if (quantityLayers != 0):
      auxLayer = self.layer
      for i in range(1, quantityLayers):
        nextL = Layer(auxLayer)
        auxLayer.nextL = nextL
        auxLayer = nextL
        auxLayer.label = 'hidden_' + str(i)
      auxLayer.label = 'last'


  def loading(self):
    os.system( 'clear' )
    print('| ')
    os.system( 'clear' )
    print('| ')
    os.system( 'clear' )
    print('| ')
    os.system( 'clear' )
    print('| ')

    os.system( 'clear' )
    print('/ ')
    os.system( 'clear' )
    print('/ ')
    os.system( 'clear' )
    print('/ ')
    os.system( 'clear' )
    print('/ ')

    os.system( 'clear' )
    print('__ ')
    os.system( 'clear' )
    print('__ ')
    os.system( 'clear' )
    print('__ ')
    os.system( 'clear' )
    print('__ ')

    os.system( 'clear' )
    print('\\ ')
    os.system( 'clear' )
    print('\\ ')
    os.system( 'clear' )
    print('\\ ')
    os.system( 'clear' )
    print('\\ ')


  # wine, flower or glass
  def set_type(self, typeAI):
    self.typeAI = typeAI


  #==========================
  #==   ENCODE OUTPUT      ==
  #==========================
  def label_encode(self, label):
    encoded = []
    if self.typeAI == 'flower':
      if label == "Iris-setosa":
        encoded = [1,0,0]
      elif label == "Iris-versicolor":
        encoded = [0,1,0]
      elif label == "Iris-virginica":
        encoded = [0,0,1]
    elif self.typeAI == 'wine':
      if label == '0':
        encoded = [0,0,0,0]
      elif label == '1':
        encoded = [0,0,0,1]
      elif label == '2':
        encoded = [0,0,1,0]
      elif label == '3':
        encoded = [0,0,1,1]
      elif label == '4':
        encoded = [0,1,0,0]
      elif label == '5':
        encoded = [0,1,0,1]
      elif label == '6':
        encoded = [0,1,1,0]
      elif label == '7':
        encoded = [0,1,1,1]
      elif label == '8':
        encoded = [1,0,0,0]
      elif label == '9':
        encoded = [1,0,0,1]
      elif label == '10':
        encoded = [1,0,1,0]
    elif self.typeAI == 'glass':
      encoded = 'n implementado'
    return encoded


  #==========================
  #==   DECODE OUTPUT      ==
  #==========================
  def data_decode(self, output):
    label = None
    if self.typeAI == 'flower':
      if output == [1,0,0]:
        label = "Iris-setosa"
      elif output == [0,1,0]:
        label = "Iris-versicolor"
      elif output == [0,0,1]:
        label = "Iris-virginica"
    elif self.typeAI == 'wine':
      if output == [0,0,0,0]:
        label = "0"
      elif output == [0,0,0,1]:
        label = "1"
      elif output == [0,0,1,0]:
        label = "2"
      elif output == [0,0,1,1]:
        label = "3"
      elif output == [0,1,0,0]:
        label = "4"
      elif output == [0,1,0,1]:
        label = "5"
      elif output == [0,1,1,0]:
        label = "6"
      elif output == [0,1,1,1]:
        label = "7"
      elif output == [1,0,0,0]:
        label = "8"
      elif output == [1,0,0,1]:
        label = "9"
      elif output == [1,0,1,0]:
        label = "10"
    elif self.typeAI == 'glass':
      label = 'n implementado'
    return label

  #==========================
  #==      SIGMÓIDE        ==
  #==========================
  def signal(self, outputNeuronium):
    return 1 / (1 + math.e**(-1 * self.beta * outputNeuronium))


  #==========================
  #==  DERIVATE SIGMÓIDE   ==
  #==========================
  def line_signal(self, outputNeuronium):
    return self.beta * self.signal(outputNeuronium) * (1 - self.signal(outputNeuronium))
    

  #==========================
  #==   POS PROCESSOR      ==
  #==========================
  def pos_signal(self, output):
    return 1 if output >= 0.5 else 0
    

  #==========================
  #==        EQM           ==
  #==========================
  def eqmCalculate(self, neuroniums, expectedOutput):
    eqm = 0
    expectedOutputEncoded = self.label_encode(expectedOutput)
    # print('saida esperada: ', expectedOutputEncoded)
    for i in range(len(neuroniums)):
      # calculate local eqm
      eqm += (expectedOutputEncoded[i] - self.signal(neuroniums[i].output)) ** 2
    
    return eqm / 2.0


  #==========================
  #==     FIX WEIGTH       ==
  #==========================
  def fix(self, neuronium, gradient):
    for iw in range(len(neuronium.w)): # update neuronium's weight
      step_momentum = self.momentum * (neuronium.w[iw] - neuronium.old_w[iw])
      # print(step_momentum)
      neuronium.w[iw] += step_momentum + self.nLearning * gradient * float(neuronium.inputs[iw])


  #==========================
  #==        FOWARD        ==
  #==========================
  def forward(self, x):
    layer = self.layer
    inputs = x[:]
    inputs.insert(0, -1) # -1 (bias) em x0
    inputsLen = len(inputs)
    outputLayer = []
    while layer is not None:
      outputLayer = []
      for i in range(len(layer.neuroniums)):
        neuronium = layer.neuroniums[i]
        u = 0
        for iw in range(inputsLen): # u = sum(inputs * weight)
          u += (neuronium.w[iw] * float(inputs[iw]))
        neuronium.inputs = inputs
        neuronium.output = u # I

        outputLayer.append(self.signal(u)) # g(I)

      inputs = outputLayer[:] # Y -> to the next layer
      if layer.nextL is not None:
        inputs.insert(0, -1) # -1 (bias) em Y0

      inputsLen = len(inputs)
      layer = layer.nextL
    # .................................................
    return outputLayer


  #==========================
  #==       BACKWARD       ==
  #==========================
  def backward(self, expectedOutput):
    layer = self.layer
    while layer.nextL is not None: # go to last layer
      layer = layer.nextL 

    expectedOutputEncoded = self.label_encode(expectedOutput)

    deltas = []
    weights = []
    while layer is not None: # go to begin
      if layer.nextL is None: # last layer
        for i in range(len(layer.neuroniums)):
          ik = 0 # index of the gradient's array & input
          gradient = (expectedOutputEncoded[i] - self.signal(layer.neuroniums[i].output)) * self.line_signal(layer.neuroniums[i].output)
          ik += 1
          deltas.append(gradient) # to next layer hidden or first
          
          old_w = layer.neuroniums[i].w[:]
          self.fix(layer.neuroniums[i], gradient) # update neuronium's weight
          layer.neuroniums[i].old_w = old_w # save weigths to calc momentum
          weights.append(layer.neuroniums[i].w) # to next layer hidden or first
      else:
        deltas_aux = []
        weights_aux = []
        for neuronium in layer.neuroniums:
          gradient = 0
          for ik in range(len(deltas)):
            for ikw in range(len(weights[ik])):
              gradient += deltas[ik] * weights[ik][ikw] 
          gradient = (gradient * self.line_signal(neuronium.output))
          deltas_aux.append(gradient) # aux to next layer hidden or first
        
          old_w = neuronium.w[:]
          self.fix(neuronium, gradient) # update neuronium's weight
          neuronium.old_w = old_w # save weigths to calc momentum
          weights_aux.append(neuronium.w) # aux to next layer hidden or first

        deltas = deltas_aux[:] # to next layer hidden or first
        weights = weights_aux[:] # to next layer hidden or first

      layer = layer.previousL 


  #==========================
  #==       TRAINING       ==
  #==========================
  def perceptro_m_c_fit(self, x, expectedOutput):
    epoch = 0
    
    # create topology
    layer = self.layer
    layerIndex = 0
    inputLen = len(x[0]) + 1 # + -1 (bias)
    while layer is not None:
      for i in range(self.neuroniumsLayer[layerIndex]):
        layer.neuroniums.append(Neuronium(inputLen)) # Create the weights randomly
      inputLen = self.neuroniumsLayer[layerIndex] + 1 # + -1 (bias)
      layerIndex += 1
      layer = layer.nextL

    # phases
    eqms = []
    eqmOld = float('inf')
    eqmCurrent = 0
    while (abs(eqmCurrent - eqmOld) > self.precision or epoch < 400): # and epoch < self.maxEpoch
      eqmOld = eqmCurrent

      layer = self.layer
      while layer.nextL is not None: # go to last layer
        layer = layer.nextL
      
      output = []
      eqm = 0
      for index, inp in enumerate(x):
        outputTopology = self.forward(inp)
        eqm += self.eqmCalculate(layer.neuroniums, expectedOutput[index])
        self.backward(expectedOutput[index])

        #saída test
        # outputProcessed = [self.pos_signal(y) for y in outputTopology]
        # output.append(self.data_decode(outputProcessed))
        # print ('u m n', outputTopology, outputProcessed, self.data_decode(outputProcessed))
      eqm /= len(x)
      eqmCurrent = eqm
      eqms.append(eqmCurrent)
      
      epoch += 1

      # Logging
      # print('Erro Médio: ', eqmCurrent)
      # print('Erro Médio old: ', eqmOld)
      # print('Erro diff: ', abs(eqmCurrent - eqmOld))
      # print('épocas: ', epoch)
      # print('layer: ', layer)

      if (epoch == self.maxEpoch):
        print(f'EXTRAPOLOU {self.maxEpoch} ÉPOCAS')

    return epoch, eqms


  #==========================
  #==      PREDICT         ==
  #==========================
  def perceptron_m_c_predict(self, x, expectedOutput):
    outputSystem_predict = []
    for i in range(len(x)):
      outputTopology = self.forward(x[i])
      outputProcessed = [self.pos_signal(y) for y in outputTopology]
      # print('saída bruta: ', outputProcessed)
      outputSystem_predict.append(self.data_decode(outputProcessed))
    
    return outputSystem_predict
