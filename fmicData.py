from numpy import *
from scipy.integrate import simps

class FmicData():

  def __init__(self, firn):
    """
    Data structure to hold intercomparison project data.
    """
    self.firn   = firn
    self.spy    = firn.const.spy
    self.rhoi   = firn.const.rhoi
    self.rhoc   = firn.const.rhoc
    self.zs     = firn.z[-1]

    self.t      = array([0.0])
    self.a      = append(0.0, firn.a/self.spy)
    self.z      = append(0.0, firn.z - self.zs)
    self.rho    = append(0.0, firn.rho)
    self.T      = append(0.0, firn.T)
    self.drhodt = append(0.0, firn.drhodt)
    self.porAll = firn.porAll
    self.por815 = firn.por815
    self.z815   = firn.z815
    self.age815 = firn.age815

  def __call__(self):
    """
    Reset the initial conditons to that of the firn object.
    """
    self.zs     = self.firn.z[-1]
    self.t      = array([0.0])
    self.a      = append(0.0, self.firn.a/self.spy)
    self.z      = append(0.0, self.firn.z - self.zs)
    self.rho    = append(0.0, self.firn.rho)
    self.T      = append(0.0, self.firn.T)
    self.drhodt = append(0.0, self.firn.drhodt)
    self.porAll = self.firn.porAll
    self.por815 = self.firn.por815
    self.z815   = self.firn.z815
    self.age815 = self.firn.age815


  def calc_fmic_variables(self):
    """
    porosity and calculations around rho = 815.  updates the firn object's data.
    """
    spy      = self.spy
    rhoi     = self.rhoi
    rhoc     = self.rhoc
    firn     = self.firn

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
  
    # interpolated depth and age :
    z815     = zb + ztot * (drhop/drhotot)
    age815   = ageb + agetot * (drhop/drhotot)
    
    # integral of porosity of whole column via simpson's rule :
    porAll   = simps(1 - firn.rho/rhoi, firn.z)

    # integral of porosity to rho = 815 :
    por815   = simps(1 - firn.rho[rho815m]/rhoi, firn.z[rho815m]) + \
               simps(1 - array([rhoa, rhoc])/rhoi, dx = za - z815)

    # update object :
    firn.porAll = porAll
    firn.por815 = por815
    firn.z815   = z815
    firn.age815 = age815


  def append_state(self, t):
    """
    append the state of firn to arrays. rows = space, cols = time.
    """
    firn = self.firn

    self.a      = vstack((self.a,      append(t, firn.a/self.spy)))
    self.z      = vstack((self.z,      append(t, firn.z - self.zs)))
    self.rho    = vstack((self.rho,    append(t, firn.rho)))
    self.T      = vstack((self.T,      append(t, firn.T)))
    self.drhodt = vstack((self.drhodt, append(t, firn.drhodt)))
  
  
  def append_815(self, t):
    """
    append porosity and rho815 values.
    """
    firn = self.firn

    self.t      = append(self.t,      t)
    self.porAll = append(self.porAll, firn.porAll)
    self.por815 = append(self.por815, firn.por815)
    self.z815   = append(self.z815,   firn.z815 - self.zs)
    self.age815 = append(self.age815, firn.age815/self.spy)
  
  
  def save_state(self, ex):
    """
    Save the current state of firn object to /data/enthalpy directory.
    """
    firn = self.firn
    ex   = str(ex)

    savetxt("data/fmic/initial" + ex + "/z.txt",   firn.z)
    savetxt("data/fmic/initial" + ex + "/l.txt",   firn.l)
    savetxt("data/fmic/initial" + ex + "/a.txt",   firn.a)
    savetxt("data/fmic/initial" + ex + "/rho.txt", firn.rho)
    savetxt("data/fmic/initial" + ex + "/H.txt",   firn.H)
    savetxt("data/fmic/initial" + ex + "/w.txt",   firn.w)
    print "saved the current state of firn"


  def save_fmic_data(self, exp):
    """
    saves the current state of this object to txt files in /data/fmic directory.
    
    INPUT    - exp: experiment number
    OUTPUTS  - txt files in fmic/exp directory
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
    por    = vstack((t, por815, porAll))
    
    directory = 'data/fmic/CummingsExperiment' + exp + '/'
  
    savetxt(directory + 'CummingsExperiment' + exp + 'Age.txt',         a,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'Depth.txt',       z,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'Density.txt',     rho,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'Temperature.txt', T,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'DensityRate.txt', drhodt,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'Porosity.txt',    por,
            delimiter='\t')
    savetxt(directory + 'CummingsExperiment' + exp + 'Rho815.txt',      rho815,
            delimiter='\t')



