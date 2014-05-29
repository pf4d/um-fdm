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
from pylab import *
from scipy.interpolate import interp1d
from dolfin import *


class Firn():
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, Tavg, rhoin, rhos, adot, dt):
    """
    """
    self.Tavg  = Tavg
    self.rhoin = rhoin
    self.rhos  = rhos
    self.adot  = adot
    self.dt    = dt


  def set_geometry(self, sur, bed):
    """
    """
    self.sur = sur
    self.bed = bed

  def generate_uniform_mesh(self, n):
    """
    """
    mesh  = IntervalMesh(n, self.bed, self.sur)  # interval from sur to bed
    z     = mesh.coordinates()[:,0]              # z-coordinates
    index = argsort(z)                           # ordered z-coord index
    
    self.mesh  = mesh
    self.z     = z
    self.index = index

    self.refine_mesh(divs=3, i=1/3.,  k=1/4.)
    self.refine_mesh(divs=1, i=1/10., k=1/4.)

  def set_parameters(self, params):
    """
    """
    self.params = params

  def set_mesh(self, mesh):
    """
    """
    self.mesh = mesh
    z         = mesh.coordinates()[:,0]          # z-coordinates
    index     = argsort(z)                       # ordered z-coord index
    
  def set_boundary_conditions(self, H_exp, rho_exp, w_exp):
    """
    """
    # enthalpy surface condition :
    self.H_S   = H_exp
    
    # density surface condition :
    self.rho_S = rho_exp

    # velocity surface condition :
    self.w_S   = w_exp

    # age surface condition (always zero at surface) :
    self.age_S = Constant(0.0)

  def refine_mesh(self, divs, i, k,  m=1):
    """
    splits the mesh a <divs> times.
  
    INPUTS:
      divs - number of times to split mesh
      i    - fraction of the mesh from the surface to split
      k    - multiple to decrease i by each step to reduce the distance from the
             surface to split
      m    - counter used to keep track of calls
  
    """
    mesh  = self.mesh
    z     = mesh.coordinates()[:,0]
    index = argsort(z)
  
    if m > divs :
      z1         = z[index]           # z-coord with correct ordering
      l          = np.diff(z1)        # distance between nodes
      self.n     = len(z)             # new number of nodes
      self.l     = l                  # height vector 
      self.index = index              # index of ordered refined mesh
      self.mesh  = mesh               # save the mesh
      self.z     = z
  
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
      self.mesh = mesh
  
      return self.refine_mesh(divs, k/i, k, m=m+1)

  def initialize_variables(self):
    """
    Initializes the class's variables to default values that are then set
    by the individually created model.
    """
    self.params.globalize_parameters(self) # make all the variables available 
    
    rhoin = self.rhoin
    n     = self.n
    cp    = self.cpi
    T0    = self.T0
    rhoi  = self.rhoi
    rhow  = self.rhow
    adot  = self.adot
    spy   = self.spy
    kcHh  = self.kcHh
    Tavg  = self.Tavg

    # create function spaces :
    V      = FunctionSpace(self.mesh, 'Lagrange', 1) # function space
    MV     = MixedFunctionSpace([V, V, V])           # mixed function space
    
    # surface Dirichlet boundary :
    def surface(x, on_boundary):
      return on_boundary and x[0] == self.sur
    
    # base Dirichlet boundary :
    def base(x, on_boundary):
      return on_boundary and x[0] == self.bed
    
    HBc   = DirichletBC(MV.sub(0), self.H_S,   surface)   # enthalpy of surface
    rhoBc = DirichletBC(MV.sub(1), self.rho_S, surface)   # density of surface
    wBc   = DirichletBC(MV.sub(2), self.w_S,   surface)   # velocity of surface
    ageBc = DirichletBC(V,         self.age_S, surface)   # age of surface
    
    #===========================================================================
    # Define variational problem spaces :
    H_i    = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
    rho_i  = interpolate(Constant(rhoin), V)          # initial density vector
    a_i    = interpolate(Constant(1.0), V)            # initial age vector
    w_i    = interpolate(Constant(0.0), V)            # initial velocity vector
    m      = interpolate(Constant(0.0), V)            # mesh velocity
    m_1    = interpolate(Constant(0.0), V)            # prev. mesh velocity
    
    epi             = Function(MV)
    h               = Function(MV)          # solution
    H, rho, w       = split(h)              # solutions for H, rho
    h_1             = Function(MV)          # previous solution
    H_1, rho_1, w_1 = split(h_1)            # initial value functions
    
    dh              = TrialFunction(MV)     # trial function for solution
    dH, drho, dw    = split(dh)             # trial functions for H, rho
    j               = TestFunction(MV)      # test function in mixed space
    psi, phi, eta   = split(j)              # test functions for H, rho
    
    a    = Function(V)                      # age solution / trial function
    da   = TrialFunction(V)                 # trial function for age
    xi   = TestFunction(V)                  # age test function
    a_1  = Function(V)                      # previous age solution
    
    epi.vector().set_local(ones(3*n))
    h_0 = project(as_vector([H_i,rho_i,w_i]), MV) # project inital values
    h.vector().set_local(h_0.vector().array())    # initalize H, rho in solution
    h_1.vector().set_local(h_0.vector().array())  # initalize in prev. sol
    
    a.vector().set_local(a_i.vector().array())    # initialize age in solution
    a_1.vector().set_local(a_i.vector().array())  # initialize age in prev. sol
    
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    Ta      = interpolate(Constant(Tavg), V)
    c       = cp
    k       = 2.1*(rho / rhoi)**2                          # Arthern 2008
    Tcoef   = interpolate(Constant(1.0), V)                # T above Tw = 0.0
    Kcoef   = interpolate(Constant(1.0), V)                # enthalpy coef.
    rhoCoef = interpolate(Constant(kcHh), V)               # density coef.
    T       = Tcoef * H / c                                # temperature
    drhodt  = Function(V)
    
    #===========================================================================
    self.V       = V                         # function space
    self.MV      = MV                        # Mixed function space
    
    self.epi     = epi
    
    self.dh      = dh                        # trial function for solution
    self.dH      = dH                        # trial function for H
    self.drho    = drho                      # trial function for rho
    self.dw      = dw                        # trial function for w
    self.j       = j                         # test function in mixed space
    self.psi     = psi                       # test function for H
    self.phi     = phi                       # test function for rho
    self.eta     = eta                       # test function for w
     
    self.da      = da                        # trial function for age
    self.xi      = xi                        # age test function
    
    self.H_i     = H_i                       # initial enthalpy
    self.rho_i   = rho_i                     # initial density
    self.w_i     = w_i                       # initial velocity
    self.a_i     = a_i                       # initial age
    self.h       = h                         # enthalpy, density, velocity
    self.H       = H                         # enthalpy
    self.H_1     = H_1                       # previous enthalpy
    self.T       = T                         # temperature
    self.rho     = rho                       # density
    self.rho_1   = rho_1                     # previous density
    self.drhodt  = drhodt                    # densification rate
    self.w       = w                         # velocity
    self.w_1     = w_1                       # previous velocity
    self.m       = m                         # mesh velocity
    self.m_1     = m_1                       # previous mesh velocity
    self.a       = a                         # age
    self.h_1     = h_1                       # previous step's solution
    self.a_1     = a_1                       # previous step's age
    self.k       = k                         # thermal conductivity
    self.c       = c                         # heat capacity
    self.bdot    = bdot                      # average annual accumulation
    self.Ta      = Ta                        # average surface temperature
    self.Tcoef   = Tcoef                     # T above Tw = 0.0 coefficient
    self.Kcoef   = Kcoef                     # enthalpy ceofficient
    self.rhoCoef = rhoCoef                   # density ceofficient

    self.Hp      = project(H, V).vector().array()[::-1]
    self.Tp      = project(T, V).vector().array()[::-1]
    self.rhop    = project(rho, V).vector().array()[::-1]
    self.drhodtp = project(drhodt, V).vector().array()[::-1]
    self.ap      = a.vector().array()[::-1]
    self.wp      = project(w, V).vector().array()[::-1]
    self.kp      = project(k, V).vector().array()[::-1]
    self.cp      = project(c, V).vector().array()[::-1]
    self.rhoinp  = self.rhop                 # initial density
    self.omega   = zeros(n)                  # water content percent
    self.agep    = zeros(n)                  # initial age
    
    self.HBc     = HBc                       # enthalpy b.c.
    self.rhoBc   = rhoBc                     # density b.c.
    self.wBc     = wBc                       # velocity b.c.
    self.ageBc   = ageBc                     # age b.c.

    self.A       = rhoi/rhow * 1e3 * adot    # surface accumulation .... mm/a
    self.lini    = self.l                    # initial height vector
    self.t       = 0.0                       # initialize time
    
    self.zb      = self.bed                  # base of firn
    self.zs      = self.sur                  # surface of firn
    self.zs_1    = self.sur                  # previous time-step surface  
    self.zo      = self.sur                  # z-coordinate of initial surface
    self.ht      = [self.zs]                 # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights
    self.Ts      = self.Hp[-1] / self.cp[-1] # temperature of surface

  
  def update_Hbc(self): 
    """
    Adjust the enthalpy at the surface.
    """
    self.H_S.t      = self.t
    self.H_S.c      = self.cp[-1]
    
  
  def update_rhoBc(self):
    """
    Adjust the density at the surface.
    """
    self.rho_S.rhoi = self.rhop[-1]
    if self.Ts > Tw:
      if self.domega[-1] > 0:
        if self.rho_S.rhon < self.rhoi:
          self.rho_S.rhon = self.rho_S.rhon + self.domega[-1]*self.rhow
      else:
        self.rho_S.rhon = self.rho_S.rhon + self.domega[-1]*83.0
    else:
      self.rho_S.rhon = self.rhos
    ltop      = lnew[-1]
    dnew      = -self.w[-1]*dt
    self.rho_S.dp = dnew/ltop
    self.rho_S.Ts = self.Tp[-1]


  def update_vars(self, t):
    """
    Project the variables onto the space V and update firn object.
    """
    self.t       = t
    adot         = self.adot

    self.Hp      = project(self.H, self.V).vector().array()[::-1]
    self.Tp      = project(self.T, self.V).vector().array()[::-1]
    self.rhop    = project(self.rho, self.V).vector().array()[::-1]
    self.drhodtp = project(self.drhodt, self.V).vector().array()[::-1]
    self.ap      = self.a.vector().array()[::-1]
    self.wp      = project(self.w, self.V).vector().array()[::-1]
    #self.kp      = project(self.k, self.V).vector().array()[::-1]
    #self.cp      = project(self.c, self.V).vector().array()[::-1]
    
    self.Ts     = self.Hp[-1] / self.cp[-1]
    self.A      = self.rhoi/self.rhow * 1e3 * adot


  def update_height_history(self):
    """
    track the current height of the firn :
    """
    self.ht.append(self.z[-1])

    # calculate the new height of original surface by interpolating the 
    # vertical speed from w and keeping the ratio intact :
    interp  = interp1d(self.z, self.wp,
                       bounds_error=False,
                       fill_value=self.wp[0])
    zint    = array([self.zo])
    wzo     = interp(zint)[0]
    dt      = self.dt
    zs      = self.z[-1]
    zb      = self.z[0]
    zs_1    = self.zs_1
    zo      = self.zo
    self.zo = zo * (zs - zb) / (zs_1 - zb) + wzo * dt
    
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
    zOld   = self.mesh.coordinates()[:,0][self.index]
    lnew   = append(0, self.lini) * self.rhoin / self.rhop
    zSum   = self.zb
    zNew   = zeros(self.n)
    for i in range(self.n):
      zNew[i]  = zSum + lnew[i]
      zSum    += lnew[i]
    self.z = zNew
    self.l = lnew[1:]
    self.mp = (zNew - zOld) / self.dt
    self.m.vector().set_local(self.mp)                 # update the mesh vel.
    self.mesh.coordinates()[:,0][self.index] = self.z  # update the mesh coord.
    self.mesh.bounding_box_tree().build(self.mesh)     # rebuild the mesh tree


  def adjust_vectors(self):
    """
    Adjust the vectors for enthalpy and density.
    """
    n    = self.n
    Tw   = self.Tw
    T0   = self.T0
    Lf   = self.Lf
    rhow = self.rhow
    kcHh = self.kcHh
    kcLw = self.kcLw
    Hsp  = self.Hsp

    # find vector of T, rho :
    self.Hp     = project(self.H, self.V).vector().array()
    self.rhop   = project(self.rho, self.V).vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    rhoCoefNew          = ones(n)
    rhoHigh             = where(self.rhop >  550)[0]
    rhoLow              = where(self.rhop <= 550)[0]
    rhoCoefNew[rhoHigh] = kcHh * (2.366 - 0.293*ln(self.A))
    rhoCoefNew[rhoLow]  = kcLw * (1.435 - 0.151*ln(self.A))
    self.rhoCoef.vector().set_local(rhoCoefNew)
  
    # update coefficients used by enthalpy :
    Hhigh               = where(self.Hp >= Hsp)[0]
    Hlow                = where(self.Hp <  Hsp)[0]
    omegaNew            = zeros(n)
    TcoefNew            = ones(n)
    KcoefNew            = ones(n)
  
    KcoefNew[Hhigh]     = 1/10.0
    TcoefNew[Hhigh]     = self.cp[Hhigh] / self.Hp[Hhigh] * Tw
  
    # update water content and density :
    omegaNew[Hhigh]     = (self.Hp[Hhigh] - self.cp[Hhigh]*(Tw - T0)) / Lf
    domega              = omegaNew - self.omega          # water content chg.
    domPos              = where(domega >  0)[0]          # water content inc.
    domNeg              = where(domega <= 0)[0]          # water content dec.
    rhoNotLiq           = where(self.rho < rhow)[0]      # density < water
    rhoInc              = intersect1d(domPos, rhoNotLiq) # where rho can inc.
    self.rhop[rhoInc]   = self.rhop[rhoInc] + domega[rhoInc]*rhow 
    self.rhop[domNeg]   = self.rhop[domNeg] + domega[domNeg]*83.0

    print self.H, self.rho_i, self.w
  
    # update the dolfin vectors :
    self.rho_i.vector().set_local(self.rhop)
    h_0 = project(as_vector([self.H, self.rho_i, self.w]), self.MV)
    self.h.vector().set_local(h_0.vector().array())
    self.Kcoef.vector().set_local(KcoefNew)
    self.Tcoef.vector().set_local(TcoefNew)
    
    self.domega = domega


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



