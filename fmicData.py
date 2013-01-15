from numpy import *

class FmicData():

  def __init__(self, firn):
    """
    Data structure to hold intercomparison project data.
    """
    self.spy    = 31556926.0
    self.rhoi   = 917.
    self.rhoc   = 815.

    self.t      = array([0.0])
    self.a      = append(0.0, firn.a)
    self.z      = append(0.0, firn.z)
    self.rho    = append(0.0, firn.rho)
    self.T      = append(0.0, firn.T)
    self.drhodt = append(0.0, firn.drhodt)
    self.porAll = firn.porAll
    self.por815 = firn.por815
    self.z815   = firn.z815
    self.age815 = firn.age815

  def __call__(self, firn):
    """
    Data structure to hold intercomparison project data.
    """
    self.t      = array([0.0])
    self.a      = append(0.0, firn.a)
    self.z      = append(0.0, firn.z)
    self.rho    = append(0.0, firn.rho)
    self.T      = append(0.0, firn.T)
    self.drhodt = append(0.0, firn.drhodt)
    self.porAll = firn.porAll
    self.por815 = firn.por815
    self.z815   = firn.z815
    self.age815 = firn.age815


  def calc_fmic_variables(self, firn):
    """
    porosity and calculations around rho=815.  updates the firn object's data.
    """
    spy      = self.spy
    rhoi     = self.rhoi
    rhoc     = self.rhoc

    rho815p  = where(firn.rho >  rhoc)[0]
    rho815m  = where(firn.rho <= rhoc)[0]
    ia       = rho815m[0]
    ib       = rho815p[-1]
  
    za       = firn.z[ia]
    zb       = firn.z[ib]
    ztot     = za - zb
  
    rhoa     = firn.rho[ia]
    rhob     = firn.rho[ib]
  
    agea     = firn.a[ia]
    ageb     = firn.a[ib]
    agetot   = agea - ageb
  
    drhom    = rhoc - rhoa
    drhop    = rhob - rhoc
    drhotot  = rhob - rhoa
  
    porAll   = sum(1 - firn.rho/rhoi)
    por815   = sum(1 - firn.rho[rho815m]/rhoi) + (1 - drhom/rhoi)
    z815     = zb + ztot * (drhop/drhotot)
    age815   = ageb + agetot * (drhop/drhotot)
  
    firn.porAll = porAll
    firn.por815 = por815
    firn.z815   = z815
    firn.age815 = age815/spy


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
    self.t      = append(self.t,      t)
    self.porAll = append(self.porAll, firn.porAll)
    self.por815 = append(self.por815, firn.por815)
    self.z815   = append(self.z815,   firn.z815)
    self.age815 = append(self.age815, firn.age815)
  
  
  def save_state(self, firn):
    """
    Save the current state of firn object to /data/enthalpy directory.
    """
    savetxt("data/enthalpy/z.txt",   firn.z)
    savetxt("data/enthalpy/a.txt",   firn.a)
    savetxt("data/enthalpy/rho.txt", firn.rho)
    savetxt("data/enthalpy/H.txt",   firn.H)
    savetxt("data/enthalpy/w.txt",   firn.w)


  def save_fmic_data(self, exp):
    """
    saves the current state of this object to txt files in /data/fmic directory.
    
    INPUT    - exp: experiment number
    OUTPUTS  - txt files in fmic directory
    """
    exp    = str(exp)
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
  
    savetxt('data/fmic/CummingsExperiment' + exp + 'Age.txt',         a,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'Depth.txt',       z,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'Density.txt',     rho,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'Temperature.txt', T,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'DensityRate.txt', drhodt,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'Porosity.txt',    por,
            delimiter='\t')
    savetxt('data/fmic/CummingsExperiment' + exp + 'Rho815.txt',      rho815,
            delimiter='\t')




