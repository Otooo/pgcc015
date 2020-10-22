import skfuzzy as fuzz
from skfuzzy import control as ctrl
import numpy as np

class Fuzzy:

  def __init__(self, relevance): 
    self.relevance = {}
    self.rules = {}

    i = 0
    iMax = len(relevance) - 1
    for re in relevance:
      ini = relevance[re][0][0]
      end = relevance[re][len(relevance[re]) - 1][len(relevance[re][0]) - 1]
      if i == iMax:
        self.relevance[re] = ctrl.Consequent(np.linspace(ini, end, 500), re)
      else:
        self.relevance[re] = ctrl.Antecedent(np.linspace(ini, end, 500), re)
      i+=1

      self.relevance[re]['low'] = fuzz.trapmf(self.relevance[re].universe, [relevance[re][0][0], relevance[re][0][0], relevance[re][0][1], relevance[re][0][2]])
      self.relevance[re]['medium'] = fuzz.trimf(self.relevance[re].universe, relevance[re][1])
      self.relevance[re]['high'] = fuzz.trapmf(self.relevance[re].universe, [relevance[re][2][0], relevance[re][2][1], relevance[re][2][2], relevance[re][2][2]])


  #==========================
  #==      SET RULES       ==
  #==========================
  def set_rules(self):
    # 1
    self.rules['1'] = ctrl.Rule(
      self.relevance['Temperature']['low'] & self.relevance['Volume']['low'],
      self.relevance['Pressure']['low']
    )
    # 2
    self.rules['2'] = ctrl.Rule(
      self.relevance['Temperature']['medium'] & self.relevance['Volume']['low'],
      self.relevance['Pressure']['low']
    )
    # 3
    self.rules['3'] = ctrl.Rule(
      self.relevance['Temperature']['high'] & self.relevance['Volume']['low'],
      self.relevance['Pressure']['medium']
    )
    #4
    self.rules['4'] = ctrl.Rule(
      self.relevance['Temperature']['low'] & self.relevance['Volume']['medium'],
      self.relevance['Pressure']['low']
    )
    #5
    self.rules['5'] = ctrl.Rule(
      self.relevance['Temperature']['medium'] & self.relevance['Volume']['medium'],
      self.relevance['Pressure']['medium']
    )
    #6
    self.rules['6'] = ctrl.Rule(
      self.relevance['Temperature']['high'] & self.relevance['Volume']['medium'],
      self.relevance['Pressure']['high']
    )
    #7
    self.rules['7'] = ctrl.Rule(
      self.relevance['Temperature']['low'] & self.relevance['Volume']['high'],
      self.relevance['Pressure']['medium']
    )
    #8
    self.rules['8'] = ctrl.Rule(
      self.relevance['Temperature']['medium'] & self.relevance['Volume']['high'],
      self.relevance['Pressure']['high']
    )
    #9
    self.rules['9'] = ctrl.Rule(
      self.relevance['Temperature']['high'] & self.relevance['Volume']['high'],
      self.relevance['Pressure']['high']
    )


  #==========================
  #==   VIEW ALL UNIVERSE  ==
  #==========================
  def relevances(self, output_simulator=None):
    if output_simulator is None:
      for re in self.relevance:
        self.relevance[re].view()
    else:
      for re in self.relevance:
        self.relevance[re].view(sim=output_simulator)


  #==========================
  #==   CONTROL SYSTEM     ==
  #==========================
  def control(self, t, v):
    # Using rules
    output_control = ctrl.ControlSystem([
      self.rules['1'], self.rules['2'], self.rules['3'], self.rules['4'], self.rules['5'], self.rules['6'], self.rules['7'], self.rules['8'], self.rules['9']
    ])
    output_simulator = ctrl.ControlSystemSimulation(output_control)

    # Inputs
    output_simulator.input['Temperature'] = t
    output_simulator.input['Volume'] = v

    # Compute output system
    output_simulator.compute()
    # Centroid value
    outputNumber = output_simulator.output['Pressure']

    return outputNumber, output_simulator
