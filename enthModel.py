"""
enthModel.py
Evan Cummings
07.19.12

FEniCS solution to firn enthalpy / density profile.

"""

from numpy import *
import numpy as np
from density import *
from dolfin import *
from scipy.interpolate import interp1d
from enthPlot import *
import sys


#===============================================================================
# constants :
pi    = 3.141592653589793      # pi
g     = 9.81                   # gravitational acceleration ..... m/s^2
R     = 8.3144621              # gas constant ................... J/(mol K)
spy   = 31556926.0             # seconds per year ............... s/a
rhoi  = 917.                   # density of ice ................. kg/m^3
rhosi = 400.                   # initial density at surface ..... kg/m^3
rhow  = 1000.                  # density of water ............... kg/m^3
rhom  = 550.                   # density at 15 m ................ kg/m^3
acc   = 91.8 / spy             # surface accumulation ........... kg/(m^2 s)
A     = spy*acc/rhosi*1e3      # surface accumulation ........... mm/a
Va    = 6.64                   # mean annual wind speed ......... m/s
ki    = 2.1                    # thermal conductivity of ice .... W/(m K)
Tw    = 273.15                 # triple point water ............. degrees K
kcHh  = 3.7e-9                 # creep coefficient high ......... (m^3 s)/kg
kcLw  = 9.2e-9                 # creep coefficient low .......... (m^3 s)/kg
kg    = 1.3e-7                 # grain growth coefficient ....... m^2/s  
Ec    = 60e3                   # act. energy for water in ice ... J/mol
Eg    = 42.4e3                 # act. energy for grain growth ... J/mol

# model variables :
n     = 80                     # num of z-positions
freq  = 2*pi/spy               # frequency of earth rotations ... rad / s
Tavg  = Tw - 5.0               # average temperature ............ degrees K
cp    = 146.3 + 7.253*Tavg     # heat capacity of ice ........... J/(kg K)
cp    = 2009.
zs    = 50.                    # surface start .................. m
zs_0  = zs                     # previous time-step surface ..... m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.025*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s

# enthalpy-specific :
T0    = 0.0                    # reference temperature .......... K
beta  = 7.9e-8                 # clausius-Clapeyron ............. K/Pa
Lf    = 3.34e5                 # latent heat of fusion .......... J/kg
Hsp   = cp*(Tw - T0)           # Enthalpy of ice at Tw .......... J/kg
omega = zeros(n+1)


#===============================================================================
# create mesh and define function space :
mesh  = Interval(n, zb, zs)

# refine mesh :
cell_markers = CellFunction("bool", mesh)
cell_markers.set_all(False)
origin = Point(zs)
for cell in cells(mesh):
  p  = cell.midpoint()
  if p.distance(origin) < 5:
    cell_markers[cell] = True
mesh = refine(mesh, cell_markers)

# update coordinates :
z      = mesh.coordinates()[:,0].copy()       # initial z-coord
numNew = len(z) - len(l)                      # number of split nodes
l      = l[:-numNew]                          # remove split heights
l      = append(l, dz/2 * ones(numNew * 2))   # append new split heights
index  = argsort(z)                           # index of updated mesh
n      = len(l)                               # new number of nodes
rhoin  = rhoi*ones(n)                         # initial density
omega  = zeros(n)                             # water content percent
z      = z[index]                             # re-order z

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV     = V*V                                  # mixed function space

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'c*( (Tavg + 9.9*sin(omega*t))  - T0)'
Hs     = Expression(code, c=cp, Tavg=Tavg, omega=freq, t=t0, T0=T0)

# temperature of base of firn :
Tb     = Constant(Tavg)

# variable surface density by S.R.M. Ligtenberg et all 2011 :
code  = '-151.94 + 1.4266*(73.6 + 1.06*Ts + 0.0669*A + 4.77*Va)'
rhoS  = Expression(code, Ts=Tavg, A=A, Va=Va)

# experimental surface density :
#code   = 'omegaS*rhow + (1 - omegaS)*rhos'
#rhoS   = Expression(code, omegaS=0.0, rhow=rhow, rhos=rhosi)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc  = DirichletBC(MV.sub(0), Hs, surface)    # enthalpy surface
Hbc2 = DirichletBC(MV.sub(0), Tb, base)       # enthalpy base 
Dbc  = DirichletBC(MV.sub(1), rhoS, surface)  # density surface


#===============================================================================
# Define variational problem :
H_i        = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
rho_i      = interpolate(Constant(rhoin[0]), V)       # initial density vector
h          = Function(MV)                    # solution
H,rho      = split(h)                        # solutions for H, rho
h_1        = Function(MV)                    # previous solution
H_0, rho_0 = split(h_1)                      # initial value functions

dh         = TrialFunction(MV)               # trial function for solution
dH, drho   = split(dh)                       # trial functions for H, rho
j          = TestFunction(MV)                # test function in mixed space
psi, phi   = split(j)                        # test functions for H, rho

h_0 = project(as_vector([H_i,rho_i]), MV)    # project inital values on space
h.vector().set_local(h_0.vector().array())   # initalize H, rho in solution
h_1.vector().set_local(h_0.vector().array()) # initalize H, rho in prev. sol


#===============================================================================
# Define equations to be solved :
w         = - acc / rho                                # vertical velocity 
w_0       = - acc / rho                                # vertical velocity 
#c         = (146.3 + sqrt(146.3**2 + 4*7.253*H)) / 2   # c in terms of H
#c         = 146.3 + 7.253*T                            # c in terms of T
c         = interpolate(Constant(cp), V)               # c constant
#k         = 9.828*exp(-0.0057*T)                      # Aschwanden 2012
k         = 2.1*(rho / rhoi)**2                        # Arthern 2008
Tcoef     = interpolate(Constant(1.0), V)
T         = Tcoef * H / c
T_0       = Tcoef * H_0 / c

Kcoef     = interpolate(Constant(1.0),  V)

f_H       = rho*(H - H_0)/dt*psi*dx + \
            k/c*Kcoef*inner(grad(H), grad(psi))*dx - \
            div(rho*H*w)*psi*dx + \
            w*grad(H)*psi*dx

# total derivative drhodt from Arthern 2010 :
rhoCoef   = interpolate(Constant(kcHh), V)
drhodt    = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T) + Eg/(R*Tavg) )*(rhoi - rho)
drho_0dt  = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T_0) + Eg/(R*Tavg) )*(rhoi - rho_0)

# material derivative :        backwards-difference :
#  dr   pr     pr               pr   r_{k} - r_{k-1}
#  -- = -- + w --               -- = ---------------
#  dt   pt     pz               pt         dt
#f_rho     = ((rho-rho_0)/dt - (drhodt - w*grad(rho)))*phi*dx

# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
#               0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 0.667 
f_rho     = ((rho-rho_0)/dt - \
            theta*(drhodt - w*grad(rho)) - \
            (1-theta)*(drho_0dt - w_0*grad(rho_0)))*phi*dx

# equation to be minimzed :
f         = f_H + f_rho
df        = derivative(f, h, dh) # jacobian


#===============================================================================
# initialize data structures :

# load initialization data :
def set_initial_converge():
  rhoin   = genfromtxt("data/enthalpy/rho.txt")
  z       = genfromtxt("data/enthalpy/z.txt")
  zs_0    = z[-1]

  rho_i.vector().set_local(rhoin)
  h_0 = project(as_vector([H_i,rho_i]), MV)    # project inital values on space
  h.vector().set_local(h_0.vector().array())   # initalize T, rho in solution
  h_1.vector().set_local(h_0.vector().array()) # initalize T, rho in prev. sol
  
  return zs_0

zs_0 = set_initial_converge()

# find vector of T, rho :
hplot   = project(H, V).vector().array()
tplot   = project(T, V).vector().array()
rhoplot = project(rho, V).vector().array()

# calculate other data :
wplot   = project(w, V).vector().array()
kplot   = project(k, V).vector().array()
cplot   = project(c, V).vector().array()

plt.ion()   # interactive mode on
firn = firn(hplot, tplot, rhoplot, omega, wplot, kplot, cplot, z, index)
plot = plot(firn)


#===============================================================================
# Compute solution :
t      = 0.0
ht     = []
origHt = []
while t <= tf:
  # newton's iterative method :
  solve(f == 0, h, [Hbc, Dbc], J=df)
  
  # find vector of T, rho :
  firn.H   = project(H, V).vector().array()
  firn.rho = project(rho, V).vector().array()
 
  # calculate other data :
  firn.w   = project(w, V).vector().array()  # m s^-1
  firn.k   = project(k, V).vector().array()  # Arthern 2008

  # calculate height of each interval (conservation of mass) :
  lnew     = l*rhoin[index] / firn.rho[index]
  zSum     = zb
  for i in range(n)[1:]:
    firn.z[i]  = zSum + lnew[i]
    zSum      += lnew[i]
  
  # correct original height with initial surface conditions :
  if t == 0.0:
    firn.origZ = firn.z[-1]
  elif t >= 5 * spy and t <= 6 * spy:
    Hs.Tavg = Tw - 10.0
  elif t >= 20 * spy:
    Hs.Tavg = Tw - 5.0

  # track the current height of the firn :
  ht.append(firn.z[-1])
  
  # track original height :
  if firn.origZ > firn.z[0]:
    origHt.append(firn.origZ)
  
  # calculate the new height of original surface by interpolating the 
  # vertical speed from w and keeping the ratio intact :
  interp      = interp1d(firn.z, firn.w[index], 
                         bounds_error=False, 
                         fill_value=firn.w[index][0])
  zint        = array([firn.origZ])
  wOrigZ      = interp(zint)
  firn.origZ  = (firn.z[-1] - zb) * (firn.origZ - zb) / (zs_0 - zb) + \
                wOrigZ[0] * dt

  # update kc term in drhodt :
  # if rho >  550, kc = kcHigh
  # if rho <= 550, kc = kcLow
  # with parameterizations given by ligtenberg et all 2011
  rhoCoefNew          = ones(n)
  rhoHigh             = where(firn.rho >  550)[0]
  rhoLow              = where(firn.rho <= 550)[0]
  rhoCoefNew[rhoHigh] = kcHh*(2.366 - 0.293*np.log(A))
  rhoCoefNew[rhoLow]  = kcLw*(1.435 - 0.151*np.log(A))
  rhoCoef.vector().set_local(rhoCoefNew)
  
  # update coefficients used by enthalpy :
  Hhigh               = where(firn.H >= Hsp)[0]
  Hlow                = where(firn.H <  Hsp)[0]
  omegaNew            = zeros(n)
  Hnew                = zeros(n)
  rhoNew              = zeros(n)
  TcoefNew            = ones(n)
  KcoefNew            = ones(n)

  KcoefNew[Hhigh]     = 1/10.0
  TcoefNew[Hhigh]     = firn.c[Hhigh] / firn.H[Hhigh] * Tw

  # update enthalpy :
  omegaNew[Hhigh]     = (firn.H[Hhigh] - firn.c[Hhigh]*(Tw - T0)) / Lf
  domega              = omegaNew - firn.omega # change in water content
  domPos              = where(domega > 0)
  domega[domPos]      = 0.0                   # make increases zero
  firn.omega          = omegaNew
  Hnew[Hhigh]         = firn.c[Hhigh]*(Tw - T0) + firn.omega[Hhigh]*Lf
  Hnew[Hlow]          = firn.H[Hlow]
  
  # update density :
  firn.rho[Hhigh]     = firn.omega[Hhigh]*rhow + \
                        200.0*domega[Hhigh] + \
                        (1 - firn.omega[Hhigh])*firn.rho[Hhigh]
 
  # update the dolfin vectors :
  rho_i.vector().set_local(firn.rho)
  H_i.vector().set_local(Hnew)
  h_0 = project(as_vector([H_i, rho_i]), MV)
  h.vector().set_local(h_0.vector().array())
  Kcoef.vector().set_local(KcoefNew)  #FIXME: erratic 
  Tcoef.vector().set_local(TcoefNew)
  
  # update firn object :
  firn.rho = project(rho, V).vector().array()
  firn.H   = project(H, V).vector().array()
  firn.T   = project(T, V).vector().array()
  firn.c   = project(c, V).vector().array()

  # update the plotting parameters :
  plot.update_plot(firn, t/spy)

  # update model parameters :
  t += dt
  h_1.assign(h)
  zs_0 = firn.z[-1]
  
  # update boundary conditions :
  Hs.t        = t
  Hs.c        = firn.c[index][-1]
  rhoS.Ts     = firn.T[index][-1]
  #rhoS.omegaS = firn.omega[index][-1]
  #rhoS.rhos   = firn.rho[index][-1]
  
  plt.draw()  # update the graph

plt.ioff()
plt.show()

# plot the surface height trend :
x = linspace(0, t/spy, len(ht))
plot.plot_height(x, ht, origHt)


