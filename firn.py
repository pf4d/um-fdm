#    Copyright (C) <2012>  <cummings.evan@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import os.path
from numpy import *
from scipy.interpolate import interp1d
from dolfin import *

class Constants():
  
  def __init__(self):
    """
    Data structure for constants.
    """
    self.pi    = 3.141592653589793 # pi ............................. rad
    self.g     = 9.81              # gravitational acceleration ..... m/s^2
    self.R     = 8.3144621         # gas constant ................... J/(mol K)
    self.spy   = 31556926.0        # seconds per year ............... s/a
    self.rhoi  = 917.              # density of ice ................. kg/m^3
    self.rhoin = self.rhoi         # initial density of column ...... kg/m^3
    self.rhow  = 1000.             # density of water ............... kg/m^3
    self.rhom  = 550.              # density at 15 m ................ kg/m^3
    self.rhoc  = 815.              # density at critical value ...... kg/m^3
    self.ki    = 2.1               # thermal conductivity of ice .... W/(m K)
    self.cpi   = 2009.             # const. heat capacitity of ice .. J/(kg K)
    self.kcHh  = 3.7e-9            # creep coefficient high ......... (m^3 s)/kg
    self.kcLw  = 9.2e-9            # creep coefficient low .......... (m^3 s)/kg
    self.kg    = 1.3e-7            # grain growth coefficient ....... m^2/s  
    self.Ec    = 60e3              # act. energy for water in ice ... J/mol
    self.Eg    = 42.4e3            # act. energy for grain growth ... J/mol
    self.Tw    = 273.15            # triple point water ............. degrees K
    self.T0    = 0.0               # reference temperature .......... K
    self.beta  = 7.9e-8            # Clausius-Clapeyron ............. K/Pa
    self.Lf    = 3.34e5            # latent heat of fusion .......... J/kg
    self.Hsp   = self.cpi*self.Tw  # Enthalpy of ice at Tw .......... J/kg


class Firn():
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, const, FEMdata, data, bcs, srf_exp, Tavg, 
               rhos, adot, A, z, l, index, dt):

    self.const   = const                     # constants

    self.mesh    = FEMdata[0]                # mesh
    self.V       = FEMdata[1]                # function space
    self.MV      = FEMdata[2]                # Mixed function space
    self.H_i     = FEMdata[3]                # initial enthalpy
    self.rho_i   = FEMdata[4]                # initial density
    self.w_i     = FEMdata[5]                # initial velocity
    self.a_i     = FEMdata[6]                # initial age
    self.h       = FEMdata[7]                # enthalpy, density, velocity
    self.HF      = FEMdata[8]                # enthalpy
    self.TF      = FEMdata[9]                # temperature
    self.rhoF    = FEMdata[10]               # density
    self.drhodtF = FEMdata[11]               # densification rate
    self.wF      = FEMdata[12]               # velocity
    self.aF      = FEMdata[13]               # age
    self.h_1     = FEMdata[14]               # previous step's solution
    self.a_1     = FEMdata[15]               # previous step's age
    self.kF      = FEMdata[16]               # thermal conductivity
    self.cF      = FEMdata[17]               # heat capacity

    self.H       = data[0]                   # enthalpy
    self.T       = data[1]                   # temperature
    self.rho     = data[2]                   # density
    self.drhodt  = data[3]                   # densification rate
    self.a       = data[4]                   # age
    self.w       = data[5]                   # vertical velocity
    self.k       = data[6]                   # thermal conductivity
    self.c       = data[7]                   # heat capacity
    self.omega   = data[8]                   # percentage of water content
    
    self.Hbc     = bcs[0]
    self.rhoBc   = bcs[1]
    self.wbc     = bcs[2]

    self.Hs      = srf_exp[0]
    self.rhoS    = srf_exp[1]
    self.wS      = srf_exp[2]

    self.Tavg    = Tavg                      # average surface temperature
    self.rhos    = rhos                      # initial density at surface
    self.adot    = adot                      # accumulation rate
    self.A       = A                         # surface accumulation
    self.z       = z[index]                  # z-coordinates of mesh
    self.l       = l                         # height vector
    self.lini    = l                         # initial height vector
    self.index   = index                     # index of ordered, refined mesh
    self.dt      = dt                        # time step
    self.t       = 0.0                       # initialize time
    
    self.n       = len(self.H)               # system DOF
    self.rhoin   = self.rho                  # initial density vector
    self.zb      = z[index][0]               # base of firn
    self.zs      = z[index][-1]              # surface of firn
    self.zs_1    = self.zs                   # previous time-step surface  
    self.zo      = self.zs                   # z-coordinate of initial surface
    self.ht      = [self.zs]                 # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights
    self.Ts      = self.H[-1] / self.c[-1]   # temperature of surface

    self.porAll  = 0.0                       # porosity of column
    self.por815  = 0.0                       # porosity to rhoc
    self.z815    = 0.0                       # depth of rhoc
    self.age815  = 0.0                       # age of rhoc

 
  def update_Hbc(self): 
    """
    Adjust the enthalpy at the surface.
    """
    self.Hs.t      = self.t
    self.Hs.c      = self.c[-1]
    
  
  def update_rhoBc(self):
    """
    Adjust the density at the surface.
    """
    self.rhoS.rhoi = self.rho[-1]
    if self.Ts > self.const.Tw:
      if domega[-1] > 0:
        if self.rhoS.rhon < self.const.rhoi:
          self.rhoS.rhon = self.rhoS.rhon + domega[-1]*self.const.rhow
      else:
        self.rhoS.rhon = self.rhoS.rhon + domega[-1]*83.0
    else:
      self.rhoS.rhon = self.rhos
    ltop      = lnew[-1]
    dnew      = -self.w[-1]*dt
    self.rhoS.dp = dnew/ltop
    self.rhoS.Ts = self.T[-1]


  def update_vars(self, t):
    """
    Project the variables onto the space V and update firn object.
    """
    self.t      = t
    rhoi        = self.const.rhoi
    rhow        = self.const.rhow
    spy         = self.const.spy
    adot        = self.adot

    self.H      = project(self.HF, self.V).vector().array()
    self.T      = project(self.TF, self.V).vector().array()
    self.rho    = project(self.rhoF, self.V).vector().array()
    self.drhodt = project(self.drhodtF, self.V).vector().array()
    self.a      = self.aF.vector().array()
    self.w      = project(self.wF, self.V).vector().array()
    #self.k      = project(self.kF, self.V).vector().array()
    #self.c      = project(self.cF, self.V).vector().array()
    
    self.Ts     = self.H[0] / self.c[0]
    self.A      = rhoi/rhow * 1e3 * adot


  def update_height_history(self):
    """
    track the current height of the firn :
    """
    self.ht.append(self.z[-1])

    # calculate the new height of original surface by interpolating the 
    # vertical speed from w and keeping the ratio intact :
    interp     = interp1d(self.z, self.w,
                          bounds_error=False,
                          fill_value=self.w[0])
    zint       = array([self.zo])
    wzo        = interp(zint)[0]
    dt         = self.dt
    zs         = self.z[-1]
    zb         = self.z[0]
    zs_1       = self.zs_1
    zo         = self.zo
    self.zo    = zo * (zs - zb) / (zs_1 - zb) + wzo * dt
    
    # track original height :
    if self.zo > zb:
      self.origHt.append(self.zo)
    
    # update the previous time steps' surface height :
    self.zs_1  = self.z[-1]


  def update_height(self):
    """
    If conserving the mass of the firn column, calculate height of each 
    interval :
    """
    lnew     = self.lini*self.rhoin / self.rho
    zSum     = self.zb
    zTemp    = zeros(self.n)
    for i in range(self.n)[1:]:
      zTemp[i] = zSum - lnew[i]
      zSum    += lnew[i]
    self.z   = zTemp
    self.l   = lnew
    self.mesh.coordinates()[:,0][self.index] = self.z  # update the mesh coord.


  def adjust_vectors(self, Kcoef, Tcoef, rhoCoef):
    """
    Adjust the vectors for enthalpy and density.
    """
    n    = self.n
    kcHh = self.const.kcHh 
    kcLw = self.const.kcLw 
    Hsp  = self.const.Hsp  
    Tw   = self.const.Tw   
    T0   = self.const.T0   
    Lf   = self.const.Lf   
    rhow = self.const.rhow

    # find vector of T, rho :
    self.H      = project(self.HF, self.V).vector().array()
    self.rho    = project(self.rhoF, self.V).vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    rhoCoefNew          = ones(n)
    rhoHigh             = where(self.rho >  550)[0]
    rhoLow              = where(self.rho <= 550)[0]
    rhoCoefNew[rhoHigh] = kcHh * (2.366 - 0.293*ln(self.A))
    rhoCoefNew[rhoLow]  = kcLw * (1.435 - 0.151*ln(self.A))
    rhoCoef.vector().set_local(rhoCoefNew)
  
    # update coefficients used by enthalpy :
    Hhigh               = where(self.H >= Hsp)[0]
    Hlow                = where(self.H <  Hsp)[0]
    omegaNew            = zeros(n)
    TcoefNew            = ones(n)
    KcoefNew            = ones(n)
  
    KcoefNew[Hhigh]     = 1/10.0
    TcoefNew[Hhigh]     = self.c[Hhigh] / self.H[Hhigh] * Tw
  
    # update water content and density :
    omegaNew[Hhigh]     = (self.H[Hhigh] - self.c[Hhigh]*(Tw - T0)) / Lf
    domega              = omegaNew - self.omega          # water content chg.
    domPos              = where(domega >  0)[0]          # water content inc.
    domNeg              = where(domega <= 0)[0]          # water content dec.
    rhoNotLiq           = where(self.rho < rhow)[0]      # density < water
    rhoInc              = intersect1d(domPos, rhoNotLiq) # where rho can inc.
    self.rho[rhoInc]    = self.rho[rhoInc] + domega[rhoInc]*rhow 
    self.rho[domNeg]    = self.rho[domNeg] + domega[domNeg]*83.0
  
    # update the dolfin vectors :
    self.rho_i.vector().set_local(self.rho)
    h_0 = project(as_vector([self.HF, self.rho_i, self.wF]), self.MV)
    self.h.vector().set_local(h_0.vector().array())
    Kcoef.vector().set_local(KcoefNew)  #FIXME: erratic 
    Tcoef.vector().set_local(TcoefNew)


  def set_ini_conv(self, ex):
    """
    sets the firn model's initial state based on files in data/enthalpy folder.
    """
    ex = str(ex)

    self.rhoin = genfromtxt("data/fmic/initial/initial" + ex + "/rho.txt")
    self.rho   = self.rhoin
    self.w     = genfromtxt("data/fmic/initial/initial" + ex + "/w.txt")
    self.z     = genfromtxt("data/fmic/initial/initial" + ex + "/z.txt")
    self.a     = genfromtxt("data/fmic/initial/initial" + ex + "/a.txt")
    self.H     = genfromtxt("data/fmic/initial/initial" + ex + "/H.txt")
    self.lin   = genfromtxt("data/fmic/initial/initial" + ex + "/l.txt")
    
    self.zs_1    = self.z[-1]                # previous time-step surface  
    self.zo      = self.z[-1]                # z-coordinate of initial surface
    self.ht      = [self.z[-1]]              # list of surface heights
    self.origHt  = [self.z[-1]]              # list of initial surface heights
    self.Ts      = self.H[-1] / self.c[-1]   # temperature of surface
  
    self.rho_i.vector().set_local(self.rho)
    self.H_i.vector().set_local(self.H)
    self.w_i.vector().set_local(self.w)
    h_0 = project(as_vector([self.H_i,self.rho_i,self.w_i]), self.MV)
    self.h.vector().set_local(h_0.vector().array())
    self.h_1.vector().set_local(h_0.vector().array())
    self.aF.vector().set_local(self.a)
    self.a_1.vector().set_local(self.a)


def refine_mesh(mesh, divs, i, k,  m=1):
  """
  splits the mesh a given number of times.

  INPUTS:
    mesh - mesh to refine
    divs - number of times to split mesh
    i    - fraction of the mesh from the surface to split
    k    - multiple to decrease i by each step to reduce the distance from the
           surface to split
    m    - counter used to keep track of calls
  OUTPUTS:
   tuple (z, l, mesh, index) - refined z-coordinates, cell height vector, 
                               mesh, and index of sorted mesh respectively

  """
  z     = mesh.coordinates()[:,0]
  index = argsort(z)

  if m > divs :
    z1    = z[index]
    z2    = z1[1:]
    z2    = append(z2, z2[-1])
    l     = z2 - z1
    return z, l, mesh, index

  else :
    zs = z[index][-1]
    zb = z[index][0]

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(zs)
    for cell in cells(mesh):
      p  = cell.midpoint()
      if p.distance(origin) < (zs - zb) * i:
        cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    return refine_mesh(mesh, divs, k/i, k, m=m+1)


def project_vars(V, H, T, rho, drhodt, a, w, k, c, omega):
  """
  Project the variables onto the space V and update firn object.
  """
  Hplot      = project(H, V).vector().array()
  Tplot      = project(T, V).vector().array()
  rhoplot    = project(rho, V).vector().array()
  drhodtplot = project(drhodt, V).vector().array()
  aplot      = a.vector().array()
  wplot      = project(w, V).vector().array()
  kplot      = project(k, V).vector().array()
  cplot      = project(c, V).vector().array() 

  return (Hplot, Tplot, rhoplot, drhodtplot, aplot, wplot, kplot, cplot, omega)


def give_density():
  """
  get the density, use like :

    s, d = give_density()
    
    x = d[0][:,0] + d[0][:,0]/(d[0][:,1] - d[0][:,0])
    x = x/100.0
    y = d[0][:,3]
    
    plot(x, y)
    
    f     = interp1d(x, y, bounds_error=False, fill_value=max(y))
    xnew  = arange(min(x), max(x)+1, 0.01)
    ynew  = f(xnew)
    
    plot(xnew, ynew, 'rx')
    
    show()  

  """
  f = open ('../../ice/data/OP60_CoreData/AllCores_Mass_Copy.txt', 'r')
  header = f.readline()

  data     = []
  stations = []
  temp     = []
  
  first = f.readline().split()
  temp.append(double(first[1:]))
  stations.append(first[0])

  for line in f.readlines():
    line    = line.split()
    station = line[0]
    dv      = double(line[1:])      # convert data values to doubles
    if station == stations[-1]:
      temp.append(dv)
    else:
      stations.append(station)
      data.append(array(temp))
      temp = []
    
  data.append(array(temp))

  return stations, array(data)



