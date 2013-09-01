"""
model.py
Evan Cummings
01.16.12

FEniCS solution to firn enthalpy / density profile.

"""

from numpy     import *
from fmicData  import *
from functions import *
from dolfin    import *
from plot      import *
import numpy as np
import sys
import time


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
n     = 100                    # num of z-positions
rhos  = 360.                   # initial density at surface ..... kg/m^3
ex    = int(sys.argv[3])

# fmic experiment :
if ex == 1:
  adot  = 0.10                 # accumulation rate .............. m/a
  Tavg  = Tw - 50.0            # average temperature ............ degrees K
elif ex == 2:
  adot  = 0.10                 # accumulation rate .............. m/a
  Tavg  = Tw - 40.0            # average temperature ............ degrees K
elif ex == 3:
  adot  = 0.10                 # accumulation rate .............. m/a
  Tavg  = Tw - 30.0            # average temperature ............ degrees K
elif ex == 4:
  adot  = 0.02                 # accumulation rate .............. m/a
  Tavg  = Tw - 30.0            # average temperature ............ degrees K
elif ex == 5:
  adot  = 0.15                 # accumulation rate .............. m/a
  Tavg  = Tw - 30.0            # average temperature ............ degrees K
elif ex == 6:
  adot  = 0.25                 # accumulation rate .............. m/a
  Tavg  = Tw - 30.0            # average temperature ............ degrees K
else :
  adot  = 0.10                 # accumulation rate .............. m/a
  Tavg  = Tw - 50.0            # average temperature ............ degrees K

acc   = rhoi * adot            # surface accumulation ........... kg/(m^2 s)
A     = acc/rhos*1e3           # surface accumulation ........... mm/a
cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
cp    = cpi                    # heat capacity of ice ........... J/(kg K)
zs    = 1000.                  # surface start .................. m
zs_0  = zs                     # previous time-step surface ..... m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.05*spy               # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s
numt  = (tf-t0)/dt             # number of time steps ........... none
times = linspace(dt,tf,numt)   # array of times to evaluate ..... s
bp    = int(sys.argv[2])       # plot or not .................... bool


#===============================================================================
# create mesh :
mesh  = IntervalMesh(n, zb, zs)
z     = mesh.coordinates()[:,0]

z, l, mesh, index = refine_mesh(mesh, divs=3, i=1/3.,  k=1/4.)
z, l, mesh, index = refine_mesh(mesh, divs=1, i=1/8.,  k=1/4.)
z, l, mesh, index = refine_mesh(mesh, divs=1, i=1/66., k=1/4.)
z, l, mesh, index = refine_mesh(mesh, divs=1, i=1/4.,  k=1/4.)
z, l, mesh, index = refine_mesh(mesh, divs=1, i=1/4.,  k=1/4.)

n      = len(l)                               # new number of nodes
rhoin  = rhoin*ones(n)                        # initial density
omega  = zeros(n)                             # water content percent
age    = zeros(n)                             # initial age

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV     = MixedFunctionSpace([V, V, V])        # mixed function space

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'c*( Tavg + 10.0*(cos(2*omega*t) + 0.3*cos(4*omega*t)))'
Hs     = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0, T0=T0)

# experimental surface density :
#code   = 'dp*rhon + (1 - dp)*rhoi'
#rhoS   = Expression(code, rhon=rhos, rhoi=rhoi, dp=1e-3)

# constant surface density :
rhoS   = Expression('rhon', rhon=rhos)

# surface age is always 0 :
ageS   = Constant(0.0)

# velocity of surface (-acc / rhos) [m/s] :
code   = '- (rhoi * adot / spy) / rhos'
wS     = Expression(code, rhoi=rhoi, adot=adot, spy=spy, rhos=rhos)

# velocity of base :
wB     = Constant(-4e-9)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc   = DirichletBC(MV.sub(0), Hs,   surface)    # enthalpy of surface
Dbc   = DirichletBC(MV.sub(1), rhoS, surface)    # density of surface
wbc   = DirichletBC(MV.sub(2), wS,   surface)    # velocity of surface
wbcb  = DirichletBC(MV.sub(2), wB,   base)       # velocity of base
ageBc = DirichletBC(V,         ageS, surface)    # age of surface


#===============================================================================
# Define variational problem spaces :
H_i        = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
rho_i      = interpolate(Constant(rhoin[0]), V)       # initial density vector
a_i        = interpolate(Constant(1.0), V)            # initial age vector
w_i        = interpolate(Constant(0.0), V)            # initial velocity vector

epi             = Function(MV)
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

epi.vector().set_local(ones(3*n))
h_0 = project(as_vector([H_i,rho_i,w_i]), MV) # project inital values on space
h.vector().set_local(h_0.vector().array())    # initalize H, rho in solution
h_1.vector().set_local(h_0.vector().array())  # initalize H, rho in prev. sol

a.vector().set_local(a_i.vector().array())    # initialize age in solution
a_1.vector().set_local(a_i.vector().array())  # initialize age in prev. sol


#===============================================================================
# Define equations to be solved :
bdot      = interpolate(Constant(rhoi * adot / spy), V)   # average annual acc
#c         = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2      # Patterson 1994
Ta        = interpolate(Constant(Tavg), V)
c         = cp
k         = 2.1*(rho / rhoi)**2                           # Arthern 2008
Tcoef     = interpolate(Constant(1.0), V)                 # T above Tw = 0.0
Kcoef     = interpolate(Constant(1.0), V)                 # enthalpy coef.
T         = Tcoef * H / c                                 # temperature

# age residual :
# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
#               0.5=Crank-Nicolson, 0=Forward-Euler) :
# uses Taylor-Galerkin upwinding :
theta     = 0.5 
a_mid     = theta*a + (1-theta)*a_1
f_a       = + (a - a_1)/dt * xi * dx \
            - 1 * xi * dx \
            + w * a_mid.dx(0) * xi * dx \
            + w**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
            - w * w.dx(0) * dt/2 * a_mid.dx(0) * xi * dx

# enthalpy residual :
theta     = 0.5
H_mid     = theta*H + (1 - theta)*H_1
f_H       = + rho * w * H_mid.dx(0) * psi * dx \
            - k/c * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
            - rho * (H - H_1)/dt * psi * dx


# density residual :
# material derivative :
#  dr   pr     pr
#  -- = -- + w --
#  dt   pt     pz
# SUPG method phihat :        
vnorm     = sqrt(dot(w, w) + 1e-10)
cellh     = CellSize(mesh)
phihat    = phi + cellh/(2*vnorm)*dot(w, phi.dx(0))

theta     = 0.878
rho_mid   = theta*rho + (1 - theta)*rho_1
rhoCoef   = interpolate(Constant(kcHh), V)
drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * (rhoi - rho_mid)
f_rho     = + (rho - rho_1)/dt * phi * dx \
            - drhodt * phihat * dx \
            + w * rho_mid.dx(0) * phihat * dx 

# velocity residual :
theta     = 0.878
w_mid     = theta*w + (1 - theta)*w_1
# Zwally equation for surface velocity :
f_w       = + rho * w_mid.dx(0) * eta * dx \
            + drhodt * eta * dx
# Arthern equation of strain rate :
f_w       = + rho**2 * w_mid.dx(0) * eta * dx \
            + bdot * rho.dx(0) * eta * dx

# equation to be minimzed :
f         = f_H + f_rho + f_w
df        = derivative(f, h, dh)   # temp/density jacobian
df_a      = derivative(f_a, a, da) # age jacobian

#===============================================================================
# initialize data structures :

# project the initial functions onto the space and initialize firn object : 
data    = project_vars(V, H, T, rho, drhodt, a, w, k, c, omega)
FEMdata = (mesh, V, MV, H_i, rho_i, w_i, a_i, h, H, T, 
           rho, drhodt, w, a, h_1, a_1, k, c)
firn    = Firn(const, FEMdata, data, Tavg, rhos, adot, A, acc, z, l, index, dt)

# load initialization data :
#firn.set_ini_conv(ex)

if bp:
  plt.ion() 
  plot = Plot(firn)
  plt.draw()
fmic = FmicData(firn)


#===============================================================================
# Compute solution :
tstart = time.clock()
set_log_active(False)
problem = NonlinearVariationalProblem(f, h, [Hbc, Dbc, wbc], J=df)
solver  = NonlinearVariationalSolver(problem)
for t in times:
  # update boundary conditions :
  Hs.t      = t
  #Hs.c      = firn.c[-1]
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
  
  #h.vector().set_local(h.vector().array() + rand())
  solver.solve()

  # newton's iterative method :
  #solve(f == 0, h, [Hbc, Dbc, wbc], J=df)

  # solve for age :
  params = {'newton_solver' : {'relaxation_parameter' : 0.90,
                               'maximum_iterations'   : 100}}
  solve(f_a == 0, a, ageBc, solver_parameters=params)
  
  # adjust the coefficient vectors :
  firn.adjust_vectors(Kcoef, Tcoef, rhoCoef)
  
  # update firn object :
  firn.update_vars()
  #firn.update_height_history()
  #firn.update_height()
  
  # update model parameters :
  h_1.assign(h)
  a_1.assign(a)

  # update the plotting parameters :
  if bp:
    plot.update_plot(firn, t/spy)

  # only start capturing the data at 5000 years :
  tr = round(t/spy,2) - 100

  # initialize the data : 
  if tr == 0.0:
    fmic.calc_fmic_variables()
    fmic()
    fmic.save_state(ex)
    print 'dt: ' + str(tr) + '\t=>\t815 SAVED'
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
  
  # update fmic 815 data :
  if tr > 0.0:
    fmic.calc_fmic_variables()
    fmic.append_815(tr)
    print 'dt: ' + str(tr) + '\t=>\t815 SAVED'
  
  # update the main fmic data:
  if tr > 0.0 and tr <= 100.0 and tr % 10 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr)
  elif tr > 100.0 and tr <= 150.0 and tr % 1 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr)
  elif tr > 150.0 and tr <= 250.0 and tr % 5 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr)
  elif tr > 250.0 and tr <= 2000.0 and tr % 10 == 0.0:
    print 'dt: ' + str(tr) + '\t=>\tSAVED'
    fmic.append_state(tr)
  elif tr < 0.0:
    print 'dt: ' + str(tr)

  
  # vary the temperature :
  if tr == 100.0 and ex == 1:
    firn.Tavg = Tw - 45.0
    Ta.vector().set_local(ones(n)*firn.Tavg)
    Hs.Tavg   = firn.Tavg
  elif tr == 100.0 and ex == 2:
    firn.Tavg = Tw - 35.0
    Ta.vector().set_local(ones(n)*firn.Tavg)
    Hs.Tavg   = firn.Tavg
  elif tr == 100.0 and ex == 3:
    firn.Tavg = Tw - 25.0
    Ta.vector().set_local(ones(n)*firn.Tavg)
    Hs.Tavg   = firn.Tavg

  # vary the accumulation :
  elif tr == 100 and ex == 4:
    firn.adot = 0.07
    bdotNew = ones(n)*(rhoi * firn.adot / spy)
    bdot.vector().set_local(bdotNew)
    wS.adot = firn.adot
  elif tr == 100 and ex == 5:
    firn.adot = 0.20  
    bdotNew = ones(n)*(rhoi * firn.adot / spy)
    bdot.vector().set_local(bdotNew)
    wS.adot = firn.adot
  elif tr == 100 and ex == 6:
    firn.adot = 0.30
    bdotNew = ones(n)*(rhoi * firn.adot / spy)
    bdot.vector().set_local(bdotNew)
    wS.adot = firn.adot
  
  # update the graph
  if bp:
    plt.draw()

# calculate total time to compute :
tfin = time.clock()
if bp:
  plt.ioff()
  plt.show()

ttot   = tfin - tstart
thours = round(ttot*(7000/tf)*spy/60/60, 3)
print "total time to process 7,000 years:", thours, "hrs"

fmic.save_fmic_data(ex)
# plot the surface height trend :
#plot.plot_height(times, firn.ht, firn.origHt)


