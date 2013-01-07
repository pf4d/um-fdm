from numpy import *

class FmicData():

  def __init__(self, firn):
    """
    Data structure to hold intercomparison project data.
    """
    self.a      = append(0.0, firn.a)
    self.z      = append(0.0, firn.z)
    self.rho    = append(0.0, firn.rho)
    self.T      = append(0.0, firn.T)
    self.drhodt = append(0.0, firn.drhodt)
    self.por    = append(0.0, firn.por)
    self.rho815 = append(0.0, firn.rho815)


  def append_state(self, t, firn):
    """
    append the state of firn to arrays. rows = space, cols = time.
    """
    self.a      = vstack((self.a,      append(t, firn.a)))
    self.z      = vstack((self.z,      append(t, firn.z)))
    self.rho    = vstack((self.rho,    append(t, firn.rho)))
    self.T      = vstack((self.T,      append(t, firn.T)))
    self.drhodt = vstack((self.drhodt, append(t, firn.drhodt)))
  
  
  def append_815(self, t, firn):
    """
    append porosity and rho815 values.
    """
    self.por    = vstack((self.por,    append(t, firn.por)))
    self.rho815 = vstack((self.rho815, append(t, firn.rho815)))
  
  
  def save_state(self):
    """
    input  - arrays
    output - txt files
    saves the current state of firn to txt files in /data/fmic directory.
    """
    a      = self.a.T
    z      = self.z.T
    rho    = self.rho.T
    T      = self.T.T
    drhodt = self.drhodt.T
    por    = self.por.T
    rho815 = self.rho815.T
  
    savetxt('data/fmic/CummingsExperiment' + exp + 'Age.txt',   a)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Depth.txt',   z)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Density.txt', rho)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Temperature.txt', T)
    savetxt('data/fmic/CummingsExperiment' + exp + 'DensityRate.txt', drhodt)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Porosity.txt', por)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Density815.txt', rho815)

