from numpy import *

class FmicData():

  def __init__(self, firn):
    """
    Data structure to hold intercomparison project data.
    """
    self.t      = array([0.0])
    self.a      = append(0.0, firn.a)
    self.z      = append(0.0, firn.z)
    self.rho    = append(0.0, firn.rho)
    self.T      = append(0.0, firn.T)
    self.drhodt = append(0.0, firn.drhodt)
    self.porAll = append(0.0, firn.porAll)
    self.por815 = append(0.0, firn.por815)
    self.z815   = append(0.0, firn.z815)
    self.age815 = append(0.0, firn.age815)


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
    self.t      = append(t,           self.t)
    self.porAll = append(self.porAll, firn.porAll)
    self.por815 = append(self.por815, firn.por815)
    self.z815   = append(self.z815,   firn.z815)
    self.age815 = append(self.age815, firn.age815)
  
  
  def save_state(self):
    """
    input  - arrays
    output - txt files
    saves the current state of firn to txt files in /data/fmic directory.
    """
    t      = self.t
    a      = self.a.T
    z      = self.z.T
    rho    = self.rho.T
    T      = self.T.T
    drhodt = self.drhodt.T
    porAll = self.porAll.T
    por815 = self.por815.T
    z815   = self.z815.T
    age815 = self.age815.T

    rho815 = vstack((t, age815, z815))
    por    = vstack((age815, por815, porAll))
  
    savetxt('data/fmic/CummingsExperiment' + exp + 'Age.txt',         a)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Depth.txt',       z)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Density.txt',     rho)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Temperature.txt', T)
    savetxt('data/fmic/CummingsExperiment' + exp + 'DensityRate.txt', drhodt)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Porosity.txt',    por)
    savetxt('data/fmic/CummingsExperiment' + exp + 'Rho815.txt',      rho815)




