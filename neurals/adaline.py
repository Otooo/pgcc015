from modules import Neuronium
from modules import Layer


class Adaline:

  def __init__(self, nLearning, precision, quantityLayers=0, maxEpoch=5000): 
    self.nLearning = nLearning
    self.precision = precision
    self.maxEpoch = maxEpoch

    self.layer = Layer()
    if (quantityLayers != 0):
      auxLayer = self.layer
      for i in range(1, quantityLayers):
        nextL = Layer(auxLayer)
        auxLayer.nextL = nextL
        auxLayer = nextL


  def signal(self, outputNeuronium):
    return 1.0 if outputNeuronium >= 0 else -1.0
  

  def fix(self, w, x, expectedOutput, outputWithoutActivation):
    return w + self.nLearning * (expectedOutput - outputWithoutActivation) * x
  

  def eqmCalculate(self, neuronium, x, expectedOutput):
    lenW = len(neuronium.w)
    eqm = 0
    for i in range(len(x)):
      u = 0
      # u <- xk * w
      for iw in range(lenW):
        u += (neuronium.w[iw] * x[i][iw])
      # calculate eqm
      eqm = eqm + (expectedOutput[i] - u) ** 2
    return eqm / len(x)


  def adaline_fit(self, x, expectedOutput):
    epoch = 0
    lenW = len(x[0])
    eqmLine = []
    neuronium = Neuronium(lenW) # Create the weights randomly
    # Logging
    print('PESOS INICIAS: ', neuronium.w)

    eqmOld = float('inf')
    eqmCurrent = 1
    while (abs(eqmCurrent - eqmOld) > self.precision and epoch < self.maxEpoch):
      eqmOld = eqmCurrent

      for i in range(len(x)):
        u = 0
        # u <- xk * w
        for iw in range(lenW):
          u += (neuronium.w[iw] * x[i][iw])
          
        # w <- w + n * (d - u) * xk
        for iw in range(lenW):
          neuronium.w[iw] = self.fix(neuronium.w[iw], x[i][iw], expectedOutput[i], u)

      eqmCurrent = self.eqmCalculate(neuronium, x, expectedOutput)
      # Errors to plot errorXepoch 
      eqmLine.append(eqmCurrent)

      epoch += 1

    if (epoch == self.maxEpoch):
      print('PARADA LIMITE ÉPOCA')

    return neuronium, epoch, eqmLine


  def adaline_predict(self, x, neuronium):
    outputPredict = []
    lenW = len(neuronium.w)
    for i in range(len(x)):
      predict = 0
      for iw in range(lenW):
        predict += (neuronium.w[iw] * x[i][iw])
      
      y = 'A' if self.signal(predict) == -1 else 'B'
      outputPredict.append(y)
    return outputPredict
