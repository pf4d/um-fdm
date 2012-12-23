"""
enthModel.py
Evan Cummings
07.19.12

FEniCS solution to firn enthalpy / density profile.

"""

from numpy import *
import numpy as np
from dolfin import *
from scipy.interpolate import interp1d
from plotFmic import *
import sys


#===============================================================================
# constants :
pi    = 3.141592653589793      # pi
g     = 9.81                   # gravitational acceleration ..... m/s^2
R     = 8.3144621              # gas constant ................... J/(mol K)
spy   = 31556926.0             # seconds per year ............... s/a
rhoi  = 917.                   # density of ice ................. kg/m^3
rhosi = 360.                   # initial density at surface ..... kg/m^3
rhow  = 1000.                  # density of water ............... kg/m^3
rhom  = 550.                   # density at 15 m ................ kg/m^3
acc   = 91.7 / spy             # surface accumulation ........... kg/(m^2 s)
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
n     = 10                     # num of z-positions
freq  = 2*pi/spy               # frequency of earth rotations ... rad / s
Tavg  = Tw - 50.0              # average temperature ............ degrees K
cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
#cp    = 2009.                 # constant heat capacitity of .... J/(kg K)
zs    = 1000.                  # surface start .................. m
zs_0  = zs                     # previous time-step surface ..... m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.5*spy                # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s

# enthalpy-specific :
T0    = 0.0                    # reference temperature .......... K
beta  = 7.9e-8                 # Clausius-Clapeyron ............. K/Pa
Lf    = 3.34e5                 # latent heat of fusion .......... J/kg
Hsp   = cp*(Tw - T0)           # Enthalpy of ice at Tw .......... J/kg
omega = zeros(n+1)


#===============================================================================
# create mesh and define function space :
mesh  = IntervalMesh(n, zb, zs)
z     = mesh.coordinates()[:,0]              # initial z-coord

def refine_mesh(mesh, z, l, dz, i, m):
  
  if m > 10 :
    return z, l, mesh
  
  else :
    zs = z[-1]
    zb = z[0]
    
    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(zs)
    for cell in cells(mesh):
      p  = cell.midpoint()
      if p.distance(origin) < (zs - zb) / i:
        cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)
    
    # update coordinates :
    z      = mesh.coordinates()[:,0]              # initial z-coord
    numNew = len(z) - len(l)                      # number of split nodes
    l      = l[:-numNew]                          # remove split heights
    l      = append(l, dz/2 * ones(numNew * 2))   # append new split heights
    
    return refine_mesh(mesh, z, l, l[-1], i*1.15, m+1)


z, l, mesh = refine_mesh(mesh, z, l, l[-1], 2, 1)

index  = argsort(z)                           # index of updated mesh
n      = len(l)                               # new number of nodes
rhoin  = rhoi*ones(n)                         # initial density
omega  = zeros(n)                             # water content percent
age    = zeros(n)                             # initial age

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV     = V*V                                  # mixed function space

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'c*( (Tavg + 10.0*(sin(2*omega*t) + 0.3*sin(4*omega*t)))  - T0 )'
Hs     = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0, T0=T0)

# temperature of base of firn :
Tb     = Constant(Tavg)

# variable surface density by S.R.M. Ligtenberg et all 2011 :
#code  = '-151.94 + 1.4266*(73.6 + 1.06*Ts + 0.0669*A + 4.77*Va)'
#rhoS  = Expression(code, Ts=Tavg, A=A, Va=Va)

# experimental surface density :
#code   = 'dp*rhon + (1 - dp)*rhoi'
#rhoS   = Expression(code, rhon=rhosi, rhoi=rhoi, dp=1e-3)

# constant surface density :
rhoS   = Expression('rhon', rhon=rhosi)

# surface age is always 0 :
ageS   = Constant(0.0)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc   = DirichletBC(MV.sub(0), Hs, surface)      # enthalpy of surface
Hbc2  = DirichletBC(MV.sub(0), Tb, base)         # enthalpy of base 
Dbc   = DirichletBC(MV.sub(1), rhoS, surface)    # density of surface
ageBc = DirichletBC(V,         ageS, surface)    # age of surface


#===============================================================================
# Define variational problem spaces :
H_i        = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
rho_i      = interpolate(Constant(rhoin[0]), V)       # initial density vector
a_i        = interpolate(Constant(1.0), V)            # initial age vector

h          = Function(MV)                    # solution
H,rho      = split(h)                        # solutions for H, rho
h_1        = Function(MV)                    # previous solution
h_2        = Function(MV)                    # 2nd previous solution
H_1, rho_1 = split(h_1)                      # initial value functions
H_2, rho_2 = split(h_2)                      # initial value functions

dh         = TrialFunction(MV)               # trial function for solution
dH, drho   = split(dh)                       # trial functions for H, rho
j          = TestFunction(MV)                # test function in mixed space
psi, phi   = split(j)                        # test functions for H, rho

a          = Function(V)                     # age solution / trial function
xi         = TestFunction(V)                 # age test function
a_1        = Function(V)                     # previous age solution
a_2        = Function(V)                     # 2nd previous age solution

h_0 = project(as_vector([H_i,rho_i]), MV)    # project inital values on space
h.vector().set_local(h_0.vector().array())   # initalize H, rho in solution
h_1.vector().set_local(h_0.vector().array()) # initalize H, rho in prev. sol
h_2.vector().set_local(h_0.vector().array()) # initalize H, rho in prev. prev.

a.vector().set_local(a_i.vector().array())   # initialize age in solution
a_1.vector().set_local(a_i.vector().array()) # initialize age in prev. sol
a_2.vector().set_local(a_i.vector().array()) # initialize age in prev. sol

#===============================================================================
# Define equations to be solved :
w         = - acc / rho                                # vertical velocity 
w_0       = - acc / rho                                # vertical velocity 
#c         = interpolate(Constant(cp), V)              # c constant
c         = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2   # Patterson 1994
#k         = 9.828*exp(-0.0057*T)                      # Aschwanden 2012
k         = 2.1*(rho / rhoi)**2                        # Arthern 2008
Tcoef     = interpolate(Constant(1.0), V)
T         = Tcoef * H / c
T_1       = Tcoef * H_1 / c

Kcoef     = interpolate(Constant(1.0),  V)

# SUPG method :        
vnorm     = sqrt(dot(w, w) + 1e-10)
cellh     = CellSize(mesh)
xihat     = xi + cellh/(2*vnorm)*dot(w, grad(xi))
phihat    = phi + cellh/(2*vnorm)*dot(w, grad(phi))

# age residual :
a_H       = ((a_2 - 4*a_1 + 3*a)/(2*dt) + w*grad(a) - 1) * xihat*dx

# enthalpy residual :
f_H       = rho*(H_2 - 4*H_1 + 3*H)/(2*dt)*psi*dx + \
            k/c*Kcoef*inner(grad(H), grad(psi))*dx + \
            rho*w*grad(H)*psi*dx

# total derivative drhodt from Arthern 2010 :
rhoCoef   = interpolate(Constant(kcHh), V)
drhodt    = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T) + Eg/(R*Tavg) )*(rhoi - rho)
drho_1dt  = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T_1) + Eg/(R*Tavg) )*(rhoi - rho_1)

# material derivative :        second difference :
#  dr   pr     pr               pr   r_{k-2} - 4*r_{k-1} + 3*r_{k}
#  -- = -- + w --               -- = -----------------------------
#  dt   pt     pz               pt                dt
#f_rho     = ((rho-rho_0)/dt - (drhodt - w*grad(rho)))*phi*dx

# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
#               0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.000

# density residual :
f_rho     = (rho_2 - 4*rho_1 + 3*rho)/(2*dt)*phi*dx - \
            theta*(drhodt - w*grad(rho))*phihat*dx - \
            (1-theta)*(drho_1dt - w_0*grad(rho_1))*phihat*dx

# equation to be minimzed :
f         = f_H + f_rho
df        = derivative(f, h, dh) # jacobian


#===============================================================================
# initialize data structures :

# load initialization data :
def set_ini_conv():
  rhoin   = genfromtxt("data/enthalpy/rho.txt")
  zTemp   = genfromtxt("data/enthalpy/z.txt")
  ain     = genfromtxt("data/enthalpy/a.txt")
  zs_0    = zTemp[-1]
  #mesh.coordinates()[:,0] = zTemp  FIXME why wouldn't this work?

  rho_i.vector().set_local(rhoin)
  a_i.vector().set_local(ain)
  h_0 = project(as_vector([H_i,rho_i]), MV)    # project inital values on space
  h.vector().set_local(h_0.vector().array())   # initalize T, rho in solution
  h_1.vector().set_local(h_0.vector().array()) # initalize T, rho in prev. sol
  h_2.vector().set_local(h_0.vector().array()) # initalize T, rho in prev. sol
  a_1.vector().set_local(ain)
  a_2.vector().set_local(ain)
  return zs_0

zs_0 = set_ini_conv()

# find vector of T, rho :
hplot   = project(H, V).vector().array()
tplot   = project(T, V).vector().array()
rhoplot = project(rho, V).vector().array()
aplot   = a.vector().array()

# calculate other data :
wplot   = project(w, V).vector().array()
kplot   = project(k, V).vector().array()
cplot   = project(c, V).vector().array()

#plt.ion()   # interactive mode on
firn = firn(hplot, tplot, rhoplot, aplot, omega, wplot, kplot, cplot, z, index)
#plot = plot(firn)


#===============================================================================
# Compute solution :
t      = 0.0
ht     = []
origHt = []
set_log_active(False)
while t <= tf:
  # newton's iterative method :
  solve(f == 0, h, [Hbc, Dbc], J=df)

  # solve for age :
  solve(a_H == 0, a, ageBc)

  # find vector of T, rho :
  firn.H   = project(H, V).vector().array()
  firn.rho = project(rho, V).vector().array()
  firn.a   = a.vector().array()
  print t/spy, '\t', min(firn.a)/spy, '\t', max(firn.a)/spy

  # calculate other data :
  firn.w   = project(w, V).vector().array()  # m s^-1
  firn.k   = project(k, V).vector().array()  # Arthern 2008

  # calculate height of each interval (conservation of mass) :
  lnew     = l*rhoin / firn.rho
  zSum     = zb
  zTemp    = zeros(n)
  for i in range(n)[1:]:
    zTemp[i] = zSum + lnew[i]
    zSum    += lnew[i]
  firn.z   = zTemp
  mesh.coordinates()[:,0][index] = firn.z

  # correct original height with initial surface conditions :
  if t == 0.0:
    firn.origZ = firn.z[-1]
    zs_0       = firn.z[-1]
  #if t >= 10 * spy:
  #  Hs.Tavg = Tw - 10.0
  #if t > 25 * spy:
  #  Hs.Tavg = Tw - 5.0
  #if t > 35 * spy:
  #  Hs.Tavg = Tw - 10.0

  # track the current height of the firn :
  ht.append(firn.z[-1])
  
  # track original height :
  if firn.origZ > firn.z[0]:
    origHt.append(firn.origZ)
  
  # calculate the new height of original surface by interpolating the 
  # vertical speed from w and keeping the ratio intact :
  interp      = interp1d(firn.z, firn.w, 
                         bounds_error=False, 
                         fill_value=firn.w[0])
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
  domega              = omegaNew - firn.omega          # water content chg.
  domPos              = where(domega >  0)[0]          # water content inc.
  domNeg              = where(domega <= 0)[0]          # water content dec.
  rhoNotLiq           = where(firn.rho < rhow)[0]      # density < water
  rhoInc              = intersect1d(domPos, rhoNotLiq) # where rho can inc.
  firn.omega          = omegaNew
  Hnew[Hhigh]         = firn.c[Hhigh]*(Tw - T0) + firn.omega[Hhigh]*Lf
  Hnew[Hlow]          = firn.H[Hlow]
  
  # update density :
  firn.rho[rhoInc]    = firn.rho[rhoInc] + domega[rhoInc]*rhow 
  firn.rho[domNeg]    = firn.rho[domNeg] + domega[domNeg]*83.0
  
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
  firn.Ts  = firn.H[-1] / firn.c[-1]

  # update the plotting parameters :
  #plot.update_plot(firn, t/spy)

  # update model parameters :
  t += dt
  h_2.assign(h_1)
  h_1.assign(h)
  a_2.assign(a_1)
  a_1.assign(a)
  zs_0 = firn.z[-1]
  
  # update boundary conditions :
  Hs.t      = t
  Hs.c      = firn.c[-1]
  #rhoS.rhoi = firn.rho[-1]
  #if firn.Ts > Tw:
  #  if domega[-1] > 0:
  #    if rhoS.rhon < rhoi:
  #      rhoS.rhon = rhoS.rhon + domega[-1]*rhow
  #  else:
  #    rhoS.rhon = rhoS.rhon + domega[-1]*83.0
  #else:
  #  rhoS.rhon = rhosi
  #ltop      = lnew[-1]
  #dnew      = -firn.w[-1]*dt
  #rhoS.dp = dnew/ltop
  #rhoS.Ts = firn.T[-1]

  #plt.draw()  # update the graph

#plot.update_plot(firn, t/spy)
#plt.draw()
#plt.ioff()
#plt.show()

savetxt('data/enthalpy/a.txt', firn.a)
savetxt('data/enthalpy/z.txt', firn.z)
savetxt('data/enthalpy/rho.txt', firn.rho)
savetxt('data/enthalpy/l.txt', l)

# plot the surface height trend :
x = linspace(0, t/spy, len(ht))
#plot.plot_height(x, ht, origHt)


