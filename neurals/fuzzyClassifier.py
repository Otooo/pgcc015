import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np
from functools import reduce
from operator import mul
import matplotlib.pyplot as plt

class FuzzyClassifier:

  #==========================
  #==    FUZZY UNIVERSE    ==
  #==========================
  @staticmethod
  def init_config():
    attrSepalLength = np.linspace(4.3, 7.92, 500)
    attrSepalWidth = np.linspace(2.0, 4.42, 500)
    attrPetalLength = np.linspace(1.0, 6.92, 500)
    attrPetalWidth = np.linspace(0.1, 2.52, 500)

    sepalLengthXMF = {}
    sepalWidthXMF = {}
    petalLengthXMF = {}
    petalWidthXMF = {}

    sepalLengthMid = (attrSepalLength[0] + attrSepalLength[-1:][0]) / 2
    sepalWidthMid = (attrSepalWidth[0] + attrSepalWidth[-1:][0]) / 2
    petalLengthMid = (attrPetalLength[0] + attrPetalLength[-1:][0]) / 2
    petalWidthMid = (attrPetalWidth[0] + attrPetalWidth[-1:][0]) / 2

    sepalLengthXMF['low']    = fuzz.trimf(attrSepalLength, [attrSepalLength[0], attrSepalLength[0], sepalLengthMid])
    sepalLengthXMF['medium'] = fuzz.trimf(attrSepalLength, [attrSepalLength[0] + (sepalLengthMid-attrSepalLength[0])/2, sepalLengthMid, sepalLengthMid+(attrSepalLength[-1:][0]-sepalLengthMid)/2])
    sepalLengthXMF['high']   = fuzz.trimf(attrSepalLength, [sepalLengthMid, attrSepalLength[-1:][0], attrSepalLength[-1:][0]])

    sepalWidthXMF['low']    = fuzz.trimf(attrSepalWidth, [attrSepalWidth[0], attrSepalWidth[0], sepalWidthMid]) 
    sepalWidthXMF['medium'] = fuzz.trimf(attrSepalWidth, [attrSepalWidth[0] + (sepalWidthMid-attrSepalWidth[0])/2, sepalWidthMid, sepalWidthMid+(attrSepalWidth[-1:][0]-sepalWidthMid)/2])
    sepalWidthXMF['high']   = fuzz.trimf(attrSepalWidth, [sepalWidthMid, attrSepalWidth[-1:][0], attrSepalWidth[-1:][0]])

    petalLengthXMF['low']    = fuzz.trimf(attrPetalLength, [attrPetalLength[0], attrPetalLength[0], petalLengthMid])
    petalLengthXMF['medium'] = fuzz.trimf(attrPetalLength, [attrPetalLength[0] + (petalLengthMid-attrPetalLength[0])/2, petalLengthMid, petalLengthMid+(attrPetalLength[-1:][0]-petalLengthMid)/2])
    petalLengthXMF['high']   = fuzz.trimf(attrPetalLength, [petalLengthMid, attrPetalLength[-1:][0], attrPetalLength[-1:][0]])

    petalWidthXMF['low']    = fuzz.trimf(attrPetalWidth, [attrPetalWidth[0], attrPetalWidth[0], petalWidthMid]) 
    petalWidthXMF['medium'] = fuzz.trimf(attrPetalWidth, [attrPetalWidth[0] + (petalWidthMid-attrPetalWidth[0])/2, petalWidthMid, petalWidthMid+(attrPetalWidth[-1:][0]-petalWidthMid)/2])
    petalWidthXMF['high']   = fuzz.trimf(attrPetalWidth, [petalWidthMid, attrPetalWidth[-1:][0], attrPetalWidth[-1:][0]])

    return attrSepalLength, attrSepalWidth, attrPetalLength, attrPetalWidth, sepalLengthXMF, sepalWidthXMF, petalLengthXMF, petalWidthXMF


  #==========================
  #==     FUZZY CHART      ==
  #==========================
  @staticmethod
  def plot(attrSepalLength, attrSepalWidth, attrPetalLength, attrPetalWidth, sepalLengthXMF, sepalWidthXMF, petalLengthXMF, petalWidthXMF):
    fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4, figsize=(8, 9))

    ax0.plot(attrSepalLength, sepalLengthXMF['low'], 'b', linewidth=1.5, label='Low')
    ax0.plot(attrSepalLength, sepalLengthXMF['medium'], 'g', linewidth=1.5, label='Medium')
    ax0.plot(attrSepalLength, sepalLengthXMF['high'], 'r', linewidth=1.5, label='High')
    ax0.set_title('Sepal Length')
    ax0.legend()

    ax1.plot(attrSepalWidth, sepalWidthXMF['low'], 'b', linewidth=1.5, label='Low')
    ax1.plot(attrSepalWidth, sepalWidthXMF['medium'], 'g', linewidth=1.5, label='Medium')
    ax1.plot(attrSepalWidth, sepalWidthXMF['high'], 'r', linewidth=1.5, label='High')
    ax1.set_title('Sepal Width')
    ax1.legend()

    ax2.plot(attrPetalLength, petalLengthXMF['low'], 'b', linewidth=1.5, label='Low')
    ax2.plot(attrPetalLength, petalLengthXMF['medium'], 'g', linewidth=1.5, label='Medium')
    ax2.plot(attrPetalLength, petalLengthXMF['high'], 'r', linewidth=1.5, label='High')
    ax2.set_title('Petal Length')
    ax2.legend()

    ax3.plot(attrPetalWidth, petalWidthXMF['low'], 'b', linewidth=1.5, label='Low')
    ax3.plot(attrPetalWidth, petalWidthXMF['medium'], 'g', linewidth=1.5, label='Medium')
    ax3.plot(attrPetalWidth, petalWidthXMF['high'], 'r', linewidth=1.5, label='High')
    ax3.set_title('Petal Width')
    ax3.legend()

    # Turn off top/right axes
    for ax in (ax0, ax1, ax2, ax3):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()


  #==========================
  #==    MAX RELEVANCE     ==
  #==========================
  @staticmethod
  def max_re(re):
    v_max = max(re)
    index = 0
    for i in range(len(re)):
      if re[i] == v_max:
        index = i

    return index, v_max


  #==========================
  #== ELIMINATE CONFLICTS  ==
  #==========================
  @staticmethod
  def eliminate_conflicts(relevances, relevance):
    include = True
    for ire in range(len(relevances)):
      if relevances[ire]['rules'] == relevance['rules']:
        include = False
        if relevances[ire]['degree'] <= relevance['degree']:
          relevances[ire] = relevance
        break

    if include:
      relevances.append(relevance)

    return relevances


  #==========================
  #==   CALCULATE PESOS    ==
  #==========================
  @staticmethod
  def calc_peso(x, y):
    (
        attrSepalLength, 
        attrSepalWidth, 
        attrPetalLength, 
        attrPetalWidth, 
        sepalLengthXMF, 
        sepalWidthXMF, 
        petalLengthXMF, 
        petalWidthXMF 
    ) = FuzzyClassifier.init_config()
    pesosCF = []

    classes = []
    for ix in range(len(x)):
      sl = {}
      sw = {}
      pl = {}
      pw = {}

      sl['low'] = fuzz.interp_membership(attrSepalLength, sepalLengthXMF['low'], x[ix][0])
      sl['medium'] = fuzz.interp_membership(attrSepalLength, sepalLengthXMF['medium'], x[ix][0])
      sl['high'] = fuzz.interp_membership(attrSepalLength, sepalLengthXMF['high'], x[ix][0])

      sw['low'] = fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['low'], x[ix][1])
      sw['medium'] = fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['medium'], x[ix][1])
      sw['high'] = fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['high'], x[ix][1])

      pl['low'] = fuzz.interp_membership(attrPetalLength, petalLengthXMF['low'], x[ix][2])
      pl['medium'] = fuzz.interp_membership(attrPetalLength, petalLengthXMF['medium'], x[ix][2])
      pl['high'] = fuzz.interp_membership(attrPetalLength, petalLengthXMF['high'], x[ix][2])

      pw['low'] = fuzz.interp_membership(attrPetalWidth, petalWidthXMF['low'], x[ix][3])
      pw['medium'] = fuzz.interp_membership(attrPetalWidth, petalWidthXMF['medium'], x[ix][3])
      pw['high'] = fuzz.interp_membership(attrPetalWidth, petalWidthXMF['high'], x[ix][3])

      i, _ = FuzzyClassifier.max_re([sl['low'], sl['medium'], sl['high']])
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      cond_sl = cond

      i, _ = FuzzyClassifier.max_re([sw['low'], sw['medium'], sw['high']])
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      cond_sw = cond

      i, _ = FuzzyClassifier.max_re([pl['low'], pl['medium'], pl['high']])
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      cond_pl = cond

      i, _ = FuzzyClassifier.max_re([pw['low'], pw['medium'], pw['high']])
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      cond_pw = cond

      classes.append({
          'sl': sl,
          'sw': sw,
          'pl': pl,
          'pw': pw,
          'cond_sl': cond_sl,
          'cond_sw': cond_sw,
          'cond_pl': cond_pl,
          'cond_pw': cond_pw,
          'class': y[ix]
      })


    for ix in range(len(classes)):
      setosaClasses = filter(lambda item: item['class'] == 'Iris-setosa', classes) 
      versicolorClasses = filter(lambda item: item['class'] == 'Iris-versicolor', classes) 
      virginicaClasses = filter(lambda item: item['class'] == 'Iris-virginica', classes) 
    
      bSetosa = sum([item['sl'][classes[ix]['cond_sl']] * item['sw'][classes[ix]['cond_sw']] * item['pl'][classes[ix]['cond_pl']] * item['pw'][classes[ix]['cond_pw']] for item in setosaClasses])
      bVersicolor = sum([item['sl'][classes[ix]['cond_sl']] * item['sw'][classes[ix]['cond_sw']] * item['pl'][classes[ix]['cond_pl']] * item['pw'][classes[ix]['cond_pw']] for item in versicolorClasses])
      bVirginica = sum([item['sl'][classes[ix]['cond_sl']] * item['sw'][classes[ix]['cond_sw']] * item['pl'][classes[ix]['cond_pl']] * item['pw'][classes[ix]['cond_pw']] for item in virginicaClasses])

      
      bClasses = [bSetosa, bVersicolor, bVirginica]
      bClassX = max(bClasses)
      for ib in range(3):
        if bClasses[ib] == bClassX:
          del bClasses[ib]
          break
      bOthers = sum(bClasses) / len(bClasses)
      
      peso = abs(bClassX - bOthers) / (sum([bSetosa, bVersicolor, bVirginica]))

      pesosCF.append(peso)

    return pesosCF


  #==========================
  #==    GENERATE RULES    ==
  #==========================
  @staticmethod
  def wang_mendel(x, y, pesosCF=None):
    (
        attrSepalLength, 
        attrSepalWidth, 
        attrPetalLength, 
        attrPetalWidth, 
        sepalLengthXMF, 
        sepalWidthXMF, 
        petalLengthXMF, 
        petalWidthXMF 
    ) = FuzzyClassifier.init_config()

    # FuzzyClassifier.plot(attrSepalLength, attrSepalWidth, attrPetalLength, attrPetalWidth, sepalLengthXMF, sepalWidthXMF, petalLengthXMF, petalWidthXMF)

    relevances = []
    for ix in range(len(x)):
      sl = []
      sw = []
      pl = []
      pw = []

      sl.append(fuzz.interp_membership(attrSepalLength, sepalLengthXMF['low'], x[ix][0]))
      sl.append(fuzz.interp_membership(attrSepalLength, sepalLengthXMF['medium'], x[ix][0]))
      sl.append(fuzz.interp_membership(attrSepalLength, sepalLengthXMF['high'], x[ix][0]))

      sw.append(fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['low'], x[ix][1]))
      sw.append(fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['medium'], x[ix][1]))
      sw.append(fuzz.interp_membership(attrSepalWidth, sepalWidthXMF['high'], x[ix][1]))

      pl.append(fuzz.interp_membership(attrPetalLength, petalLengthXMF['low'], x[ix][2]))
      pl.append(fuzz.interp_membership(attrPetalLength, petalLengthXMF['medium'], x[ix][2]))
      pl.append(fuzz.interp_membership(attrPetalLength, petalLengthXMF['high'], x[ix][2]))

      pw.append(fuzz.interp_membership(attrPetalWidth, petalWidthXMF['low'], x[ix][3]))
      pw.append(fuzz.interp_membership(attrPetalWidth, petalWidthXMF['medium'], x[ix][3]))
      pw.append(fuzz.interp_membership(attrPetalWidth, petalWidthXMF['high'], x[ix][3]))

      values = []
      rules = []
      degree = []

      rev = 'Se '
      i, v = FuzzyClassifier.max_re(sl)
      values.append(v)
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      rules.append(cond)
      rev += f'SepalLength é {cond} && '

      i, v = FuzzyClassifier.max_re(sw)
      values.append(v)
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      rules.append(cond)
      rev += f'SepalWidth é {cond} && '

      i, v = FuzzyClassifier.max_re(pl)
      values.append(v)
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      rules.append(cond)
      rev += f'PetalLength é {cond} && '

      i, v = FuzzyClassifier.max_re(pw)
      values.append(v)
      cond = 'low'
      if i == 1: cond = 'medium'
      elif i == 2: cond = 'high'
      rules.append(cond)
      rev += f'PetalWidth é {cond}, Então a classe é {y[ix]}'

      peso = reduce(mul, values, 1)
      if pesosCF is not None:
        peso *= pesosCF[ix]
      relevance = {
          'condition': rev,
          'values': values,
          'rules': rules,
          'degree': peso,
          'class': y[ix]
      }

      relevances = FuzzyClassifier.eliminate_conflicts(relevances, relevance)

    return relevances


  #==========================
  #==    INIT INSTANCE     ==
  #==========================
  def __init__(self, rules): 
    self.rules = rules
    (
        self.attrSepalLength, 
        self.attrSepalWidth, 
        self.attrPetalLength, 
        self.attrPetalWidth, 
        self.sepalLengthXMF, 
        self.sepalWidthXMF, 
        self.petalLengthXMF, 
        self.petalWidthXMF 
     ) = FuzzyClassifier.init_config()


  #==========================
  #==      FIND CLASS      ==
  #==========================
  def find_class(self, relevance):
    return relevance[1]


  #==========================
  #==  T-NORMA CALCULATE   ==
  #==========================
  def calc_norma(self, irisRules, pertN, tNormaClass, method):
    tNormaClassAux = []
    for irisRule in irisRules:
      relev = []
      relev.append(pertN['sl'][irisRule['rules'][0]])
      relev.append(pertN['sw'][irisRule['rules'][1]])
      relev.append(pertN['pl'][irisRule['rules'][2]])
      relev.append(pertN['pw'][irisRule['rules'][3]])
      tNorma = min(relev)
      if tNorma > 0:             #0        #1
        tNormaClassAux.append((tNorma, irisRule['class']))

    if 'mrfc' == method:
      if len(tNormaClassAux) > 0:
        tNormaClass.append(max(tNormaClassAux))
    elif 'mrfg' == method:
      if len(tNormaClassAux) > 0:
        mean = 0
        for normac in tNormaClassAux:
          mean += normac[0]
        mean = mean / len(tNormaClassAux)
        tNormaClass.append((mean, tNormaClassAux[0][1]))
    
    return tNormaClass


  #==========================
  #==   MRF... CLASSIFIER  ==
  #==========================
  def common_classifier(self, x, ix, method):
    sl = {}
    sw = {}
    pl = {}
    pw = {}

    sl['low'] = fuzz.interp_membership(self.attrSepalLength, self.sepalLengthXMF['low'], x[ix][0])
    sl['medium'] = fuzz.interp_membership(self.attrSepalLength, self.sepalLengthXMF['medium'], x[ix][0])
    sl['high'] = fuzz.interp_membership(self.attrSepalLength, self.sepalLengthXMF['high'], x[ix][0])

    sw['low'] = fuzz.interp_membership(self.attrSepalWidth, self.sepalWidthXMF['low'], x[ix][1])
    sw['medium'] = fuzz.interp_membership(self.attrSepalWidth, self.sepalWidthXMF['medium'], x[ix][1])
    sw['high'] = fuzz.interp_membership(self.attrSepalWidth, self.sepalWidthXMF['high'], x[ix][1])

    pl['low'] = fuzz.interp_membership(self.attrPetalLength, self.petalLengthXMF['low'], x[ix][2])
    pl['medium'] = fuzz.interp_membership(self.attrPetalLength, self.petalLengthXMF['medium'], x[ix][2])
    pl['high'] = fuzz.interp_membership(self.attrPetalLength, self.petalLengthXMF['high'], x[ix][2])

    pw['low'] = fuzz.interp_membership(self.attrPetalWidth, self.petalWidthXMF['low'], x[ix][3])
    pw['medium'] = fuzz.interp_membership(self.attrPetalWidth, self.petalWidthXMF['medium'], x[ix][3])
    pw['high'] = fuzz.interp_membership(self.attrPetalWidth, self.petalWidthXMF['high'], x[ix][3])
    
    pertN = {
        'sl': sl,
        'sw': sw,
        'pl': pl,
        'pw': pw
    }

    setosaRules = filter(lambda item: item['class'] == 'Iris-setosa', self.rules) 
    versicolorRules = filter(lambda item: item['class'] == 'Iris-versicolor', self.rules) 
    virginicaRules = filter(lambda item: item['class'] == 'Iris-virginica', self.rules) 
    tNormaClass = []

    tNormaClass = self.calc_norma(setosaRules, pertN, tNormaClass, method)
    tNormaClass = self.calc_norma(versicolorRules, pertN, tNormaClass, method)
    tNormaClass = self.calc_norma(virginicaRules, pertN, tNormaClass, method)

    return tNormaClass


  #==========================
  #==   MRFC CLASSIFIER    ==
  #==========================
  def classifier_MRFC(self, x):
    y = []
    for ix in range(len(x)):
      tNormaClass = self.common_classifier(x, ix, 'mrfc')
      
      maxClass = max(tNormaClass) if len(tNormaClass) > 0 else [0, None]
      y.append(self.find_class(maxClass))

    return y


  #==========================
  #==   MRFG CLASSIFIER    ==
  #==========================
  def classifier_MRFG(self, x):
    y = []
    for ix in range(len(x)):
      tNormaClass = self.common_classifier(x, ix, 'mrfg')
      
      maxClass = max(tNormaClass) if len(tNormaClass) > 0 else [0, None]
      y.append(self.find_class(maxClass))

    return y
