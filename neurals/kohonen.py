import math
import numpy as np
from modules import Neuronium
from copy import copy, deepcopy
from sklearn.cluster import KMeans


class Kohonen:

  def __init__(self, nLearning, radius, matrix, maxEpoch=5000): 
    self.nLearning = nLearning
    self.radius = radius
    self.maxEpoch = maxEpoch
    self.topology = np.empty( (matrix, matrix), dtype=object)


  #==========================
  #==     FIX WEIGTH       ==
  #==========================
  def fix(self, w, x, n):
    for iw in range(len(w)):
      w[iw] = w[iw] + n * (x[iw] - w[iw])

    return w


  #==========================
  #==    CALC DISTANCE     ==
  #==========================
  def euclidean_distance(self, x, w):
    distance = 0
    for i in range(len(x)):
      distance += (x[i] - w[i])**2

    return math.sqrt(distance)


  #==========================
  #==     GET WINNER       ==
  #==========================
  def competition(self, x):
    m = len(self.topology)
    matrixDistance = np.zeros((m, m))

    winnerIndex = [0, 0]
    neuroniumDistance = self.euclidean_distance(x, self.topology[0][0].w)

    for r in range(len(self.topology)):
      for c in range(len(self.topology[0])):
        distance = self.euclidean_distance(x, self.topology[r][c].w)
        matrixDistance[r][c] = distance
        if distance < neuroniumDistance:
          winnerIndex = [r, c]
          neuroniumDistance = distance
    
    return winnerIndex, matrixDistance


  #==========================
  #==     COOP & ADAPT     ==
  #==========================
  def cooperation_adaptation(self, x, winnerIndex):
    # winnerIndex[0] == row & winnerIndex[1] == column

    # winner
    self.topology[winnerIndex[0]][winnerIndex[1]].w = self.fix(self.topology[winnerIndex[0]][winnerIndex[1]].w, x, self.nLearning)
    
    # neighbors
    for rc in range(4):
      if rc & 1 == 0: # horizontal
        r_radius = winnerIndex[0]
        if rc < 2: # left
          c_radius = - self.radius + winnerIndex[1]
        else: # right
          c_radius = self.radius + winnerIndex[1]
      else: # vertical
        if rc < 2: # top
          r_radius = - self.radius + winnerIndex[0]
        else: # bottom
          r_radius = self.radius + winnerIndex[0]
        c_radius = winnerIndex[1]

      if r_radius >= 0 and r_radius < len(self.topology):
        if c_radius >= 0 and c_radius < len(self.topology[0]):
          self.topology[r_radius][c_radius].w = self.fix(self.topology[r_radius][c_radius].w, x, (self.nLearning / 2))


  #==========================
  #==     MATRIX U         ==
  #==========================
  def u_matrix(self):
    m = len(self.topology)
    uMatrix = np.zeros((m, m))

    for r in range(len(self.topology)):
      for c in range(len(self.topology[0])):
        sumD = 0
        qttNeigh = 0

        if r - 1 >= 0: # top
          sumD += self.euclidean_distance(self.topology[r-1][c].w, self.topology[r][c].w); qttNeigh += 1
        if r + 1 < len(self.topology): # bottom
          sumD += self.euclidean_distance(self.topology[r+1][c].w, self.topology[r][c].w); qttNeigh += 1
        if c - 1 >= 0: # left
          sumD += self.euclidean_distance(self.topology[r][c-1].w, self.topology[r][c].w); qttNeigh += 1
        if c + 1 < len(self.topology[0]): # right
          sumD += self.euclidean_distance(self.topology[r][c+1].w, self.topology[r][c].w); qttNeigh += 1

        uMatrix[r][c] = sumD / qttNeigh

    return uMatrix


  #==========================
  #==        FIT          ==
  #==========================
  def kohonen_fit(self, x):
    epoch = 0
    lenW = len(x[0])

    # Define neurons weights
    for r in range(len(self.topology)):
      for c in range(len(self.topology[0])):
        self.topology[r][c] = Neuronium(lenW)

    # define matrix of winners
    m = len(self.topology)
    matrixNeurons = np.zeros((m, m, lenW))

    while True and epoch < m*self.maxEpoch:
      matrixNeurons_old = copy(matrixNeurons)

      for ix in range(len(x)):
        winnerIndex, matrixDistance = self.competition(x[ix])
        self.cooperation_adaptation(x[ix], winnerIndex)

        matrixNeurons[winnerIndex[0]][winnerIndex[1]] = copy(self.topology[winnerIndex[0]][winnerIndex[1]].w)

      epoch += 1

      if (epoch % self.maxEpoch == 0):
        print(f'MARCA DE {epoch} ÉPOCAS')
        
      if (matrixNeurons_old == matrixNeurons).all():
        break
    
    return epoch, self.u_matrix(), [[n.w for n in rc] for rc in self.topology]


  #==========================
  #==      PREDICT         ==
  #==========================
  def kohonen_predict(self, x, k, outputTest):
    outputPredict = {}
    kMeans = self.k_means(k)
    
    for ik in range(k):
      outputPredict[ik] = []

    for ix in range(len(x)):
      kClassIndex = 0
      distanceK = self.euclidean_distance(x[ix], kMeans[kClassIndex])
      for ik in range(1, k):
        distanceKAux = self.euclidean_distance(x[ix], kMeans[ik])
        if distanceK > distanceKAux:
          distanceK = distanceKAux
          kClassIndex = ik

      outputPredict[kClassIndex].append(outputTest[ix])

    return outputPredict, kMeans


  #==========================
  #==      K MEANS         ==
  #==========================
  def k_means(self, k):
    ws = []
    for r in range(len(self.topology)):
      for c in range(len(self.topology[0])):
        ws.append(self.topology[r][c].w)

    kmeans = KMeans(n_clusters=k, init='k-means++').fit(ws)
    return kmeans.cluster_centers_
