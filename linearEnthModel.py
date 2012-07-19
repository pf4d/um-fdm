"""
linearEnthModel.py
Evan Cummings
05.23.12

FEniCS solution to firn linear enthalpy profile.

"""

from numpy import *
import numpy as np
from density import *
from dolfin import *
from scipy.interpolate import interp1d
from enthPlot import *
import sys


#==============================================================================
# constants :
pi    = 3.141592653589793      # pi
g     = 9.81                   # gravitational acceleration ..... m/s^2
R     = 8.3144621              # gas constant ................... J/(mol K)
spy   = 31556926.0             # seconds per year ............... s/a
rhoi  = 917.                   # density of ice ................. kg/m^3
rhosi = 300.                   # initial density at surface ..... kg/m^3
rhow  = 1000.                  # density of water ............... kg/m^3
rhom  = 550.                   # density at 15 m ................ kg/m^3
acc   = 250. / spy             # surface accumulation ........... kg/(m^2 s)
A     = spy*acc/rhosi          # surface accumulation ........... m/a
Va    = 10.                    # mean annual wind speed ......... m/s
cp    = 2009.                  # heat capacity of ice ........... J/(kg K)
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
zs    = 50.                    # surface start .................. m
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


#==============================================================================
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
z      = mesh.coordinates()[:,0].copy()             # initial z-coord
numNew = len(z) - len(l)                            # number of split nodes
l      = l[:-numNew]                                # remove split heights
l      = append(l, dz/2 * ones(numNew * 2))         # append new split heights
index  = argsort(z)                                 # index of updated mesh
rhoin  = rhoi*ones(len(l))                          # initial density
omega  = zeros(len(l))
z      = z[index]

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)  

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'ci*( (Tavg + 9.9*sin(omega*t))  - T0)'
Hs     = Expression(code, ci=cp, Tavg=Tavg, omega=freq, t=0.0, T0=T0)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc    = DirichletBC(V, Hs, surface)    # temperature surface


#==============================================================================
# Define variational problem :
H_1 = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
H   = TrialFunction(V)                         # trial function for solution
phi = TestFunction(V)                          # test function in mixed space


#==============================================================================
# Define equations to be solved :
# expression for variable rate constant KT :
#K  = 8.36*T**(-2.061)                          # Reeh ZL correction
#dK = 8.36*-2.061*T**(-1.061)                   # derivative K(T)

# thermal conductivity Van Dusen formula (lower limit) :
#k  = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# thermal conductivity Schwerdtfeger forumla (upper limit) :
#k  = (2*ki*rho) / (3*rhoi - rho)

# thermal conductivity Arthern et all 1998 :
#k  = 2.1*(rho / rhoi)**2

# thermal conductivity Greve and Blatter 2009 :
#k  = 9.828*exp(-0.0057 * T)

k  = ki*ones(len(l))
w  = -acc / rhoin

# heat capacity Aschwanden 2012 - Greve and Blatter 2009 :
#ci = 146.3 + 7.253 * T

# pressure :
p  = Expression('rhoi * g * x[0]', rhoi=rhoi, g=g)

Kcoef = interpolate(Constant(ki/cp),  V)

a  = rhoi*H*phi*dx + dt*Kcoef*inner(grad(H), grad(phi))*dx 
L  = rhoi*H_1*phi*dx

A  = assemble(a)
b  = None
H  = Function(V)


#==============================================================================
# initialize data structures :
# find vector of H, T:
hplot   = H.vector().array()
tplot   = hplot / cp

plt.ion()
firn = firn(hplot, tplot, rhoin, omega, w, k, z, index, zb, zs)
plot = plot(firn)


#==============================================================================
# Compute solution :
t      = 0.0
ht     = []
origHt = []
while t <= tf:
  # update boundary conditions :
  Hs.t     = t
 
  # solve that shit : 
  b = assemble(L, tensor=b)
  Hbc.apply(A, b)
  solve(A, H.vector(), b)
  
  # update state of firn :
  firn.H   = H.vector().array()
  firn.T   = firn.H / cp
  firn.rho = rhoin
 
  # calculate other data :
  firn.w   = -acc / firn.rho # m s^-1
  firn.k   = (2*ki*firn.rho) / (3*rhoi - firn.rho)
 
  # calculate height of each interval (conservation of mass) :
  lnew     = l*rhoin[index] / firn.rho[index]
  zSum     = zb
  for i in range(len(z))[1:]: 
    firn.z[i]  = zSum + lnew[i]
    zSum      += lnew[i]
  
  # correct original height with initial surface conditions :
  if t == 0.0:
    firn.origZ = firn.z[-1]
  
  # track the current height and original surface height of the firn :
  ht.append(firn.z[-1])
  origHt.append(firn.origZ)
  
  # calculate the new height of original surface
  # by interpolating vertical speed from w :
  #if firn.origZ > firn.z[0]:
  #  interp      = interp1d(firn.z, firn.w[index])
  #  zint        = array([firn.origZ])
  #  wOrigZ      = interp(zint)
  #  firn.origZ += wOrigZ[0] * dt
  #else:
  #  firn.origZ  = 0.0

  # update coefficients and stuff :
  Hhigh                = where(firn.H >= Hsp)[0]
  Hlow                 = where(firn.H <  Hsp)[0]
  omegaNew             = zeros(len(firn.T))
  Hnew                 = zeros(len(firn.H))
  Tnew                 = zeros(len(firn.T))
  KcoefNew             = zeros(len(firn.T))
  omegaNew[Hhigh]      = (firn.H[Hhigh] - Hsp) / Lf
  omegaNew[Hlow]       = 0.0
  Tnew[Hhigh]          = Tw
  Tnew[Hlow]           = firn.T[Hlow]
  Hnew[Hhigh]          = Hsp + omega[Hhigh]*Lf
  Hnew[Hlow]           = firn.H[Hlow]
  KcoefNew[Hhigh]      = ki/(cp*10)
  KcoefNew[Hlow]       = ki/cp
  firn.omega           = omegaNew
  firn.T               = Tnew
  Kcoef.vector().set_local(KcoefNew)
  H.vector().set_local(Hnew)
  
  # update the plotting parameters :
  plot.update_plot(t/spy)
  
  plt.draw()  # update the graph

  # update time and previous solution :
  t += dt
  H_1.assign(H)
  
plt.ioff()
plt.show()

# plot the surface height trend :
#x = linspace(0, t/spy, len(ht))
#plot.plot_height(x, ht, origHt)


