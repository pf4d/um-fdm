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
from pylab             import *
from scipy.interpolate import interp1d
from fenics            import *
from ufl.indexed       import Indexed
from termcolor         import colored


class Firn(object):
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
    self.S = sur
    self.B = bed

  def generate_uniform_mesh(self, n):
    """
    """
    mesh  = IntervalMesh(n, self.B, self.S)      # interval from bed to surface
    z     = mesh.coordinates()[:,0]              # z-coordinates
    index = argsort(z)                           # ordered z-coord index
    
    self.mesh  = mesh
    self.z     = z[index]
    self.index = index

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
    S     = self.S
    B     = self.B
  
    if m < divs :
      cell_markers = CellFunction("bool", mesh)
      cell_markers.set_all(False)
      origin = Point(S)
      for cell in cells(mesh):
        p  = cell.midpoint()
        if p.distance(origin) < (S - B) * i:
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

    self.z     = self.mesh.coordinates()[:,0]
    self.index = argsort(self.z)
    self.z     = self.z[self.index]
    self.l     = np.diff(self.z)
    self.n     = len(self.z)
   
    index = self.index 
    rhoin = self.rhoin
    n     = self.n
    cpi   = self.cpi
    T0    = self.T0
    rhoi  = self.rhoi
    rhow  = self.rhow
    adot  = self.adot
    spy   = self.spy
    kcHh  = self.kcHh
    Tavg  = self.Tavg
    Hsp   = self.Hsp

    # create function spaces :
    V      = FunctionSpace(self.mesh, 'Lagrange', 1) # function space
    
    # surface Dirichlet boundary :
    def surface(x, on_boundary):
      return on_boundary and x[0] == self.S
    
    # base Dirichlet boundary :
    def base(x, on_boundary):
      return on_boundary and x[0] == self.B
    
    HBc   = DirichletBC(V, self.H_S,   surface)   # enthalpy of surface
    rhoBc = DirichletBC(V, self.rho_S, surface)   # density of surface
    wBc   = DirichletBC(V, self.w_S,   surface)   # velocity of surface
    ageBc = DirichletBC(V, self.age_S, surface)   # age of surface
    
    #===========================================================================
    # Define variational problem spaces :
    H_i    = interpolate(Constant(cpi*(Tavg - T0)), V) # initial enthalpy vector
    rho_i  = interpolate(Constant(rhoin), V)           # initial density vector
    a_i    = interpolate(Constant(1.0), V)             # initial age vector
    w_i    = interpolate(Constant(0.0), V)             # initial velocity vector
    m      = interpolate(Constant(0.0), V)             # mesh velocity
    m_1    = interpolate(Constant(0.0), V)             # prev. mesh velocity
    
    H      = Function(V)
    rho    = Function(V)
    w      = Function(V)                              # solutions for H, rho
   
    H_1    = Function(V)
    rho_1  = Function(V)
    w_1    = Function(V)
    
    dH     = TrialFunction(V)
    drho   = TrialFunction(V)
    dw     = TrialFunction(V)
    
    psi    = TestFunction(V) 
    phi    = TestFunction(V) 
    eta    = TestFunction(V) 
    
    a      = Function(V)                      # age solution / trial function
    da     = TrialFunction(V)                 # trial function for age
    xi     = TestFunction(V)                  # age test function
    a_1    = Function(V)                      # previous age solution
   
    self.assign_variable(H,     H_i)
    self.assign_variable(H_1,   H_i)
    self.assign_variable(rho,   rho_i)
    self.assign_variable(rho_1, rho_i)
    self.assign_variable(w,     w_i)
    self.assign_variable(w_1,   w_i)
    self.assign_variable(a,     a_i)
    self.assign_variable(a_1,   a_i)
    
    T       = interpolate(Constant(Tavg), V)
    omega   = interpolate(Constant(0.0), V)
    omega_1 = interpolate(Constant(0.0), V)
    drhodt  = Function(V)
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    Ta      = interpolate(Constant(Tavg), V)
    c       = interpolate(Constant(cpi), V)
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    k       = 2.1*(rho / rhoi)**2                          # Arthern 2008
    Kcoef   = interpolate(Constant(1.0), V)                # enthalpy coef.
    #Kcoef   = conditional( lt(H, Hsp), 1.0, 1.0/10.0 )     # enthalpy coef.
    rhoCoef = interpolate(Constant(kcHh), V)               # density coef.

    #===========================================================================
    self.V       = V                         # function space

    self.dt_v    = Constant(self.dt)         # timestep Constant
    
    self.dH      = dH                        # trial function for H
    self.drho    = drho                      # trial function for rho
    self.dw      = dw                        # trial function for w
    self.psi     = psi                       # test function for H
    self.phi     = phi                       # test function for rho
    self.eta     = eta                       # test function for w
     
    self.da      = da                        # trial function for age
    self.xi      = xi                        # age test function
    
    self.H_i     = H_i                       # initial enthalpy
    self.rho_i   = rho_i                     # initial density
    self.w_i     = w_i                       # initial velocity
    self.a_i     = a_i                       # initial age
    self.H       = H                         # enthalpy
    self.H_1     = H_1                       # previous enthalpy
    self.T       = T                         # temperature
    self.domega  = zeros(self.n)             # change in water content
    self.omega   = omega                     # water content
    self.omega_1 = omega                     # water content
    self.rho     = rho                       # density
    self.rho_1   = rho_1                     # previous density
    self.drhodt  = drhodt                    # densification rate
    self.w       = w                         # velocity
    self.w_1     = w_1                       # previous velocity
    self.m       = m                         # mesh velocity
    self.m_1     = m_1                       # previous mesh velocity
    self.a       = a                         # age
    self.a_1     = a_1                       # previous step's age
    self.k       = k                         # thermal conductivity
    self.c       = c                         # heat capacity
    self.bdot    = bdot                      # average annual accumulation
    self.Ta      = Ta                        # average surface temperature
    self.Kcoef   = Kcoef                     # enthalpy ceofficient
    self.rhoCoef = rhoCoef                   # density ceofficient

    self.Hp      = H.vector().array()[index]
    self.Tp      = T.vector().array()[index]
    self.omegap  = omega.vector().array()    # water content percent
    self.rhop    = rho.vector().array()[index]
    self.drhodtp = project(drhodt, V).vector().array()[index]
    self.ap      = a.vector().array()[index]
    self.wp      = w.vector().array()[index]
    self.kp      = project(k,V).vector().array()[index]
    self.cp      = project(c,V).vector().array()[index]
    self.rhoinp  = self.rhop                 # initial density
    self.agep    = zeros(n)                  # initial age
    
    self.HBc     = HBc                       # enthalpy b.c.
    self.rhoBc   = rhoBc                     # density b.c.
    self.wBc     = wBc                       # velocity b.c.
    self.ageBc   = ageBc                     # age b.c.

    self.A       = rhoi/rhow * 1e3 * adot    # surface accumulation .... mm/a
    self.lini    = self.l                    # initial height vector
    self.lnew    = self.l.copy()             # previous height vector
    self.t       = 0.0                       # initialize time
    
    self.S_1     = self.S                    # previous time-step surface  
    self.zo      = self.S                    # z-coordinate of initial surface
    self.ht      = [self.S]                  # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights
    self.Ts      = self.Hp[-1] / self.cp[-1] # temperature of surface
  
  def assign_variable(self, u, var):
    """
    Manually assign the values from <var> to Function <u>.  <var> may be an
    array, float, Expression, or Function.
    """
    if isinstance(u, Indexed):
      u = project(u, self.Q)

    if   isinstance(var, float) or isinstance(var, int):
      u.vector()[:] = var
    
    elif isinstance(var, np.ndarray):
      u.vector().set_local(var)
      u.vector().apply('insert')
    
    elif isinstance(var, Expression):
      u.interpolate(var)

    elif isinstance(var, GenericVector):
      u.vector().set_local(var.array())
      u.vector().apply('insert')

    elif isinstance(var, Function):
      u.vector().set_local(var.vector().array())
      u.vector().apply('insert')
    
    elif isinstance(var, Indexed):
      u.vector().set_local(project(var, self.Q).vector().array())
      u.vector().apply('insert')

    elif isinstance(var, str):
      File(var) >> u

    else:
      print "*************************************************************"
      print "assign_variable() function requires a Function, array, float," + \
            " int, \nVector, Expression, Indexed, or string path to .xml, " + \
            "not \n%s" % type(var)
      print "*************************************************************"
      exit(1)
  
  def update_Hbc(self): 
    """
    Adjust the enthalpy at the surface.
    """
    self.H_S.t = self.t
    self.H_S.c = self.cp[-1]

  def update_rhoBc(self):
    """
    Adjust the density at the surface.
    """
    self.rho_S.t = self.t
  
  def update_wBc(self):
    """
    Adjust the velocity at the surface.
    """
    self.w_S.t    = self.t
    self.w_S.rhos = self.rhop[-1]
    bdotNew       = (self.w_S.adot * self.rhoi) / self.spy
    self.assign_variable(self.bdot, bdotNew)

  
  #def update_rhoBc(self):
  #  """
  #  Adjust the density at the surface.
  #  """
  #  domega_s = self.domega[self.index][-1]
  #  if self.Ts > self.Tw:
  #    if domega_s >= 0:
  #      if self.rho_S.rhon < self.rhoi:
  #        self.rho_S.rhon += domega_s*self.rhow
  #    else:
  #      self.rho_S.rhon += domega_s*self.rhow#83.0
  #  else:
  #    self.rho_S.rhon = self.rhos


  def update_vars(self, t):
    """
    Project the variables onto the space V and update firn object.
    """
    self.t       = t
    V            = self.V
    adot         = self.adot
    index        = self.index

    self.Hp      = self.H.vector().array()[index]
    self.rhop    = self.rho.vector().array()[index]
    self.wp      = self.w.vector().array()[index]
    self.ap      = self.a.vector().array()[index]
    self.Tp      = self.T.vector().array()[index]
    self.omegap  = self.omega.vector().array()[index]
    #self.drhodtp = project(self.drhodt, self.V).vector().array()[index]
    #self.kp      = project(self.k, self.V).vector().array()[index]
    #self.cp      = project(self.c, self.V).vector().array()[index]
    
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
    wzo     = interp(self.zo)
    dt      = self.dt
    zs      = self.z[-1]
    zb      = self.z[0]
    zs_1    = self.S_1
    zo      = self.zo
    #self.zo = zo * (zs - zb) / (zs_1 - zb) + wzo * dt
    self.zo = zo + (zs - zs_1) + wzo * dt
    
    # track original height :
    if self.zo > zb:
      self.origHt.append(self.zo)
    
    # update the previous time steps' surface height :
    self.S_1  = self.z[-1]

  def update_height(self):
    """
    If conserving the mass of the firn column, calculate height of each 
    interval :
    """
    zOld   = self.z
    lnew   = append(0, self.lini) * self.rhoin / self.rhop
    zSum   = self.B
    zNew   = zeros(self.n)
    for i in range(self.n):
      zNew[i]  = zSum + lnew[i]
      zSum    += lnew[i]
    self.z    = zNew
    self.l    = lnew[1:]
    self.mp   = -(zNew - zOld) / self.dt
    self.lnew = lnew
    
    self.assign_variable(self.m_1, self.m)
    self.assign_variable(self.m,   self.mp)
    self.mesh.coordinates()[:,0][self.index] = self.z  # update the mesh coord.
    self.mesh.bounding_box_tree().build(self.mesh)     # rebuild the mesh tree

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
    
    self.S_1    = self.z[-1]                # previous time-step surface  
    self.zo     = self.z[-1]                # z-coordinate of initial surface
    self.ht     = [self.z[-1]]              # list of surface heights
    self.origHt = [self.z[-1]]              # list of initial surface heights
    self.Ts     = self.H[-1] / self.c[-1]   # temperature of surface
  
    self.assign_variable(self.rho_i, self.rho)
    self.assign_variable(self.H_i,   self.H)
    self.assign_variable(self.w_i,   self.w)
    self.assign_variable(self.aF,    self.a)
    self.assign_variable(self.a_1,   self.a)

  def print_min_max(self, u, title):
    """
    Print the minimum and maximum values of <u>, a Vector, Function, or array.
    """
    if isinstance(u, GenericVector):
      uMin = u.array().min()
      uMax = u.array().max()
    elif isinstance(u, Function):
      uMin = u.vector().array().min()
      uMax = u.vector().array().max()
    elif isinstance(u, np.ndarray):
      uMin = u.min()
      uMax = u.max()
    elif isinstance(u, Indexed):
      u_n  = project(u, self.Q)
      uMin = u_n.vector().array().min()
      uMax = u_n.vector().array().max()
    else:
      print "print_min_max function requires a Vector, Function, array," \
            + " or Indexed, not %s." % type(u)
      uMin = uMax = 0.0
    s    = title + ' <min, max> : <%f, %f>' % (uMin, uMax)
    text = colored(s, 'yellow')
    print text



