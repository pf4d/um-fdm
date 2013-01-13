"""
enthModel.py
Evan Cummings
07.19.12

FEniCS solution to firn enthalpy / density profile.

"""

from numpy import *
from fmicData import *
from modelFunctions import *
import numpy as np
from dolfin import *
from plotFmic import *
import sys


#===============================================================================
# constants :

const = Constants()

pi    = const.pi               # pi
g     = const.g                # gravitational acceleration ..... m/s^2
R     = const.R                # gas constant ................... J/(mol K)
spy   = const.spy              # seconds per year ............... s/a
rhoi  = const.rhoi             # density of ice ................. kg/m^3
rhoin = const.rhoin            # initial density of column ...... kg/m^3
rhow  = const.rhow             # density of water ............... kg/m^3
rhom  = const.rhom             # density at 15 m ................ kg/m^3
rhoc  = const.rhoc             # density at critical value ...... kg/m^3
ki    = const.ki               # thermal conductivity of ice .... W/(m K)
cpi   = const.cpi              # const. heat capacitity of ice .. J/(kg K)
kcHh  = const.kcHh             # creep coefficient high ......... (m^3 s)/kg
kcLw  = const.kcLw             # creep coefficient low .......... (m^3 s)/kg
kg    = const.kg               # grain growth coefficient ....... m^2/s  
Ec    = const.Ec               # act. energy for water in ice ... J/mol
Eg    = const.Eg               # act. energy for grain growth ... J/mol
Tw    = const.Tw               # triple point water ............. degrees K
T0    = const.T0               # reference temperature .......... K
beta  = const.beta             # Clausius-Clapeyron ............. K/Pa
Lf    = const.Lf               # latent heat of fusion .......... J/kg
Hsp   = const.Hsp              # Enthalpy of ice at Tw .......... J/kg

# model variables :
n     = 75                     # num of z-positions
rhos  = 360.                   # initial density at surface ..... kg/m^3
adot  = 0.10                   # accumulation rate .............. m/a
acc   = rhoi * adot / spy      # surface accumulation ........... kg/(m^2 s)
A     = spy*acc/rhos*1e3       # surface accumulation ........... mm/a
Tavg  = Tw - 50.0              # average temperature ............ degrees K
cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
zs    = 1000.                  # surface start .................. m
zs_0  = zs                     # previous time-step surface ..... m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.020*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s


#===============================================================================
# create mesh :
mesh  = IntervalMesh(n, zb, zs)
z     = mesh.coordinates()[:,0]

z, l, mesh = refine_mesh(mesh, z, l, divs=7, dz=l[-1], i=1.5, k=1.05)

index  = argsort(z)                           # index of updated mesh
n      = len(l)                               # new number of nodes
rhoin  = rhoin*ones(n)                        # initial density
omega  = zeros(n)                             # water content percent
age    = zeros(n)                             # initial age

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV     = MixedFunctionSpace([V, V, V])        # mixed function space

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'c*( (Tavg + 10.0*(cos(2*omega*t) + 0.3*cos(4*omega*t)))  - T0 )'
Hs     = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0, T0=T0)

## simplified enthalpy surface condition :
#code   = 'c*( Tavg - T0 )'
#Hs     = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0, T0=T0)

# experimental surface density :
#code   = 'dp*rhon + (1 - dp)*rhoi'
#rhoS   = Expression(code, rhon=rhos, rhoi=rhoi, dp=1e-3)

# constant surface density :
rhoS   = Expression('rhon', rhon=rhos)

# surface age is always 0 :
ageS   = Constant(0.0)

# velocity of surface (-acc / rhos) :
code   = '- (rhoi * adot / spy) / rhos'
wS     = Expression(code, rhoi=rhoi, adot=adot, spy=spy, rhos=rhos)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc   = DirichletBC(MV.sub(0), Hs,   surface)    # enthalpy of surface
Dbc   = DirichletBC(MV.sub(1), rhoS, surface)    # density of surface
wbc   = DirichletBC(MV.sub(2), wS,   surface)    # velocity of surface
ageBc = DirichletBC(V,         ageS, surface)    # age of surface


#===============================================================================
# Define variational problem spaces :
H_i        = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
rho_i      = interpolate(Constant(rhoin[0]), V)       # initial density vector
a_i        = interpolate(Constant(1.0), V)            # initial age vector
w_i        = interpolate(Constant(acc), V)            # initial velocity vector

h               = Function(MV)                # solution
H, rho, w       = split(h)                    # solutions for H, rho
h_1             = Function(MV)                # previous solution
H_1, rho_1, w_1 = split(h_1)                  # initial value functions

dh              = TrialFunction(MV)           # trial function for solution
dH, drho, dw    = split(dh)                   # trial functions for H, rho
j               = TestFunction(MV)            # test function in mixed space
psi, phi, eta   = split(j)                    # test functions for H, rho

a          = Function(V)                      # age solution / trial function
da         = TrialFunction(V)                 # trial function for age
xi         = TestFunction(V)                  # age test function
a_1        = Function(V)                      # previous age solution

h_0 = project(as_vector([H_i,rho_i,w_i]), MV) # project inital values on space
h.vector().set_local(h_0.vector().array())    # initalize H, rho in solution
h_1.vector().set_local(h_0.vector().array())  # initalize H, rho in prev. sol

a.vector().set_local(a_i.vector().array())    # initialize age in solution
a_1.vector().set_local(a_i.vector().array())  # initialize age in prev. sol

#===============================================================================
# Define equations to be solved :
bdot      = interpolate(Constant(rhoi * adot / spy), V)   # average annual acc
c         = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2      # Patterson 1994
k         = 2.1*(rho / rhoi)**2                           # Arthern 2008
Tcoef     = interpolate(Constant(1.0), V)                 # T above Tw = 0.0
Kcoef     = interpolate(Constant(1.0),  V)                # enthalpy coef.
T         = Tcoef * H / c                                 # temperature

# age residual :
# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
#               0.5=Crank-Nicolson, 0=Forward-Euler) :
# uses Taylor-Galerkin upwinding :
theta     = 0.5 
a_mid     = theta*a + (1-theta)*a_1
f_a       = (a - a_1)/dt*xi*dx \
            - 1.*xi*dx \
            + w*grad(a_mid)*xi*dx \
            + w**2*dt/2*inner(grad(a_mid), grad(xi))*dx \
            - w*grad(w)*dt/2*grad(a_mid)*xi*dx

# enthalpy residual :
theta     = 0.5
H_mid     = theta*H + (1 - theta)*H_1
f_H       = rho*(H - H_1)/dt*psi*dx + \
            k/c*Kcoef*inner(grad(H_mid), grad(psi))*dx + \
            rho*w*grad(H_mid)*psi*dx

# density residual :
# material derivative :
#  dr   pr     pr
#  -- = -- + w --
#  dt   pt     pz
# SUPG method phihat :        
vnorm     = sqrt(dot(w, w) + 1e-10)
cellh     = CellSize(mesh)
phihat    = phi + cellh/(2*vnorm)*dot(w, grad(phi))

theta     = 1.0
rho_mid   = theta*rho + (1 - theta)*rho_1
rhoCoef   = interpolate(Constant(kcHh), V)
drhodt    = (bdot*g*rhoCoef/kg)*exp( -Ec/(R*T) + Eg/(R*Tavg) )*(rhoi - rho_mid)
f_rho     = (rho - rho_1)/dt*phi*dx - \
            (drhodt - w*grad(rho_mid))*phihat*dx 

# velocity residual :
theta     = 1.0
w_mid     = theta*w + (1 - theta)*w_1
f_w       = rho*grad(w_mid)*eta*dx + drhodt*eta*dx

# equation to be minimzed :
f         = f_H + f_rho + f_w
df        = derivative(f, h, dh)   # temp/density jacobian
df_a      = derivative(f_a, a, da) # age jacobian

#===============================================================================
# initialize data structures :

# load initialization data :
#zs_0 = set_ini_conv(H_i, rho_i, w_i, h, h_1, a, a_1)

# project the initial functions onto the space and initialize firn object : 
data    = project_vars(V, H, T, rho, drhodt, a, w, k, c, omega)
FEMdata = (mesh, V, MV, H_i, rho_i, w_i, a_i, h, H, rho, w, a, h_1, a_1)
firn    = firn(const, FEMdata, data, rhos, adot, A, acc, z, l, index, dt)

plt.ion() 
plot = plot(firn)
fmic = FmicData(firn)


#===============================================================================
# Compute solution :
t = 0.0
set_log_active(False)
while t <= tf - dt:
  # newton's iterative method :
  solve(f == 0, h, [Hbc, Dbc, wbc], J=df)

  # solve for age :
  solve(f_a == 0, a, ageBc)
  
  # update model parameters :
  t += dt
  h_1.assign(h)
  a_1.assign(a)

  # adjust the coefficient vectors :
  firn.adjust_vectors(Kcoef, Tcoef, rhoCoef)
  
  # update firn object :
  data = project_vars(V, H, T, rho, drhodt, a, w, k, c, omega)
  firn.update_firn(data)
  
  # calculate the fmic data and update the firn object :
  fmic.calc_fmic_variables(firn)

  # calculate height of each interval (conservation of mass) :
  #firn.update_height()

  # update the plotting parameters :
  plot.update_plot(firn, t/spy)
  #print t/spy, min(firn.a)/spy, max(firn.a)/spy
 
  # for modulo arithmetic :
  tr = round(t/spy, 2)
  
  # update fmic data :
  if tr % 1 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\t815 SAVED'
    fmic.append_815(tr, firn)
  
  if t <= 100.0*spy and tr % 10 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr, firn)

  elif t > 100.0*spy and t < 150.0*spy and tr % 1 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr, firn)
    
  elif t >= 150.0*spy and t < 250.0*spy and tr % 5 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr, firn)

  elif t >= 250.0*spy and t <= 2000.0*spy and tr % 10 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr, firn)
  
  # vary the temperature :
  if t == 100.0 * spy:
    Hs.Tavg = Tw - 45.0
  #if t == 100.0 * spy:
  #  Hs.Tavg = Tw - 35.0
  #if t == 100.0 * spy:
  #  Hs.Tavg = Tw - 25.0

  ## vary the accumulation :
  #if t == 100 * spy:
  #  firn.adot = 0.07
  #  #bdotNew = ones(n)*(rhoi * firn.adot / spy)
  #  #bdot.vector().set_local(bdotNew)
  #  wS.adot = firn.adot
  #if t == 100 * spy:
  #  firn.adot = 0.20  
  #  #bdotNew = ones(n)*(rhoi * firn.adot / spy)
  #  #bdot.vector().set_local(bdotNew)
  #  wS.adot = firn.adot
  #if t == 100 * spy:
  #  firn.adot = 0.30
  #  #bdotNew = ones(n)*(rhoi * firn.adot / spy)
  #  #bdot.vector().set_local(bdotNew)
  #  wS.adot = firn.adot
  
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
  #  rhoS.rhon = rhos
  #ltop      = lnew[-1]
  #dnew      = -firn.w[-1]*dt
  #rhoS.dp = dnew/ltop
  #rhoS.Ts = firn.T[-1]

  plt.draw()  # update the graph

plt.ioff()
plt.show()

# plot the surface height trend :
x = linspace(0, t/spy, len(firn.ht))
plot.plot_height(x, firn.ht, firn.origHt)


