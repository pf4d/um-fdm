"""
enthModel.py
Evan Cummings
05.23.12

FEniCS solution to firn temperature/density profile.

run with "python objModel.py <model> <end time> <initialize>

model -  :
  zl ... Li and Zwally 2002 model with Reeh correction.
  hl ... Herron and Langway 1980 [unworking]
  a .... Arthern 2008

end time -  time to run the model in years

"""

from numpy import *
import numpy as np
from density import *
from dolfin import *
from scipy.interpolate import interp1d
from plot import *
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
omega = 2*pi/spy               # frequency of earth rotations ... rad / s
Tavg  = Tw - 10.0              # average temperature ............ degrees K
zs    = 50.                    # surface start .................. m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.025*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s


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
z      = z[index]

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)         # function space for rho, T
MV     = V*V                                        # mixed function space

# cyclical surface temperature :
code  = 'Tavg + 9.9*sin(omega*t)'
Ts    = Expression(code, Tavg=Tavg, omega=omega, t=0.0)

# temperature of base of firn :
Tb    = Constant(Tavg)

# variable surface density by S.R.M. Ligtenberg et all 2011 :
code  = '-151.94 + 1.4266*(73.6 + 1.06*Ts + 0.0669*A + 4.77*Va)'
rhoS  = Expression(code, Ts=Tavg, A=A, Va=Va)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Tbc  = DirichletBC(MV.sub(0), Ts, surface)    # temperature surface
Tbc2 = DirichletBC(MV.sub(0), Tb, base)       # temperature base 
Dbc  = DirichletBC(MV.sub(1), rhoS, surface)  # density surface


#==============================================================================
# Define variational problem :
T_i        = interpolate(Constant(Tavg), V)  # initial temperature vector
rho_i      = interpolate(Constant(rhoi), V)  # initial density vector
h          = Function(MV)                    # solution
T,rho      = split(h)                        # solutions for T, rho
h_1        = Function(MV)                    # previous solution
T_0, rho_0 = split(h_1)                      # initial value functions

dh         = TrialFunction(MV)               # trial function for solution
dT, drho   = split(dh)                       # trial functions for T, rho
j          = TestFunction(MV)                # test function in mixed space
psi, phi   = split(j)                        # test functions for T, rho

h_0 = project(as_vector([T_i,rho_i]), MV)    # project inital values on space
h.vector().set_local(h_0.vector().array())   # initalize T, rho in solution
h_1.vector().set_local(h_0.vector().array()) # initalize T, rho in prev. sol


#==============================================================================
# Define equations to be solved :
# expression for vertical velocity of firn :
w         = - acc / rho
wm1       = - acc / rho_0

# expression for variable rate constant KT :
K         = 8.36*T**(-2.061)                          # Reeh ZL correction
dK        = 8.36*-2.061*T**(-1.061)                   # derivative K(T)

# thermal conductivity Van Dusen formula (lower limit) :
#k = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# thermal conductivity Schwerdtfeger forumla (upper limit) :
#k = (2*ki*rho) / (3*rhoi - rho)

# thermal conductivity Arthern et all 1998 :
#  dk    pk pr   pk pT
#  -- =  -- -- + -- --  (chain rule)
#  dz    pr pz   pT pz
k         = 2.1*(rho / rhoi)**2
dkdrho    = 4.2*(rho / rhoi**2)
drhodT    = 9.828*-5.7e-3*exp(-5.7e-3 * T)    # Patterson pg. 205
dkdT      = 4.2*(drhodT / rhoi**2)
dkdz      = dkdrho*grad(rho) + dkdT*grad(T)

f_T       = (rho*cp*(T-T_0)*psi/dt + \
            k*inner(grad(T),grad(psi)) + \
            rho*cp*w*grad(T)*psi + \
            dkdz*grad(T)*psi)*dx

# total derivative drhodt from Arthern 2010 :
rhoCoef  = interpolate(Constant(kcHh), V)
drhodt   = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T) + Eg/(R*Tavg) )*(rhoi - rho)

# material derivative :        backwards-difference :
#  dr   pr     pr               pr   r_{k} - r_{k-1}
#  -- = -- + w --               -- = ---------------
#  dt   pt     pz               pt         dt
f_rho    = ((rho-rho_0)/dt - (drhodt - w*grad(rho)))*phi*dx

# equation to be minimzed :
f  = f_T + f_rho
df = derivative(f, h, dh) # jacobian


#==============================================================================
# initialize data structures :
# find vector of T, rho :
tplot   = project(T, V).vector().array()
rhoplot = project(rho, V).vector().array()

# calculate other data :
wplot   = -acc / rhoplot * 1e3
kplot1  = 2.1e-2 + 4.2e-4*rhoplot + 2.2e-9*rhoplot**3
kplot2  = 2.1*(rhoplot / rhoi)**2
kplot3  = (2*ki*rhoplot) / (3*rhoi - rhoplot)

plt.ion()
firn = firn(tplot, rhoplot, wplot, kplot1, 
            kplot2, kplot3, z, index, zb, zs)
plot = plot(firn)


#==============================================================================
# Compute solution :
t      = 0.0
ht     = []
origHt = []
while t <= tf:
  # update boundary conditions :
  Ts.t     = t
  rhoS.Ts  = firn.T[index][-1]

  # newton's iterative method :
  solve(f == 0, h, [Tbc, Dbc], J=df)
  
  # find vector of T, rho :
  firn.T   = project(T, V).vector().array()
  firn.rho = project(rho, V).vector().array()
 
  # calculate other data :
  firn.w   = -acc / firn.rho # m s^-1
  firn.k1  = 2.1e-2 + 4.2e-4*firn.rho + 2.2e-9*firn.rho**3
  firn.k2  = 2.1*(firn.rho / rhoi)**2
  firn.k3  = (2*ki*firn.rho) / (3*rhoi - firn.rho)
 
  # calculate height of each interval (conservation of mass) :
  lnew     = l*rhoin[index] / firn.rho[index]
  zSum     = zb
  for i in range(len(z)):
    firn.z[i]  = zSum + lnew[i]
    zSum      += lnew[i]
  
  # correct original height with initial surface conditions :
  if t == 0.0:
    firn.origZ = firn.z[-1]
  
  # update the plotting parameters :
  plot.update_plot(t/spy)
  
  plt.draw()  # update the graph
  
  # track the current height and original surface height of the firn :
  ht.append(firn.z[-1])
  origHt.append(firn.origZ)
  
  # calculate the new height of original surface
  # by interpolating vertical speed from w :
  if firn.origZ > firn.z[0]:
    interp      = interp1d(firn.z, firn.w[index])
    zint        = array([firn.origZ])
    wOrigZ      = interp(zint)
    firn.origZ += wOrigZ[0] * dt
  else:
    firn.origZ  = 0.0

  # update kc term in drhodt :
  # if rho >  54, kc = kcHigh
  # if rho <= 550, kc = kcLow
  rhoCoefNew          = ones(len(firn.rho))
  rhoHigh             = where(firn.rho >  550)[0]
  rhoLow              = where(firn.rho <= 550)[0]
  rhoCoefNew[rhoHigh] = kcHh
  rhoCoefNew[rhoLow]  = kcLw
  rhoCoef.vector().set_local(rhoCoefNew)

  # update time and previous solution :
  t        += dt
  h_1.assign(h)
  
plt.ioff()
plt.show()

# plot the surface height trend :
x = linspace(0, t/spy, len(ht))
plot.plot_height(x, ht, origHt)


