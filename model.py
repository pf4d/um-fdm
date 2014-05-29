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
"""
model.py
Evan Cummings
01.16.12

FEniCS solution to firn enthalpy / density profile.

"""

from numpy     import *
from dolfin    import *
from plot      import *
from firn      import *
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

# fmic experiment :
adot  = 0.25                   # accumulation rate .............. m/a
Tavg  = Tw - 10.0              # average temperature ............ degrees K

A     = rhoi/rhow * 1e3 * adot # surface accumulation ........... mm/a
cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
cp    = cpi                    # heat capacity of ice ........... J/(kg K)
zs    = 0.                     # surface start .................. m
zs_0  = zs                     # previous time-step surface ..... m
zb    = -100.                  # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 10.0*spy               # time-step ...................... s
dt    = 0.025*spy               # time-step ...................... s
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
z, l, mesh, index = refine_mesh(mesh, divs=1, i=1/10., k=1/4.)

n      = len(z)                               # new number of nodes
rhoin  = rhoin*ones(n)                        # initial density
omega  = zeros(n)                             # water content percent
age    = zeros(n)                             # initial age

# create function spaces :
V      = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV     = MixedFunctionSpace([V, V, V])        # mixed function space

# enthalpy surface condition with cyclical 2-meter air temperature :
code   = 'c*( Tavg + 10.0*(sin(2*omega*t) + 5*sin(4*omega*t)))'
Hs     = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0, T0=T0)

# experimental surface density :
#code   = 'dp*rhon + (1 - dp)*rhoi'
#rhoS   = Expression(code, rhon=rhos, rhoi=rhoi, dp=1e-3)

# constant surface density :
rhoS   = Expression('rhon', rhon=rhos)

# surface age is always 0 :
ageS   = Constant(0.0)

# velocity of surface (-acc / rhos) [m/s] :
code   = '- rhoi/rhos * adot / spy'
wS     = Expression(code, rhoi=rhoi, adot=adot, spy=spy, rhos=rhos)

# define the Dirichlet boundarys :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def base(x, on_boundary):
  return on_boundary and x[0] == zb

Hbc   = DirichletBC(MV.sub(0), Hs,   surface)    # enthalpy of surface
rhoBc = DirichletBC(MV.sub(1), rhoS, surface)    # density of surface
wbc   = DirichletBC(MV.sub(2), wS,   surface)    # velocity of surface
ageBc = DirichletBC(V,         ageS, surface)    # age of surface
bcs   = [Hbc, rhoBc, wbc]
srf_exp = [Hs, rhoS, wS]

#===============================================================================
# Define variational problem spaces :
H_i        = interpolate(Constant(cp*(Tavg - T0)), V) # initial enthalpy vector
rho_i      = interpolate(Constant(rhoin[0]), V)       # initial density vector
a_i        = interpolate(Constant(1.0), V)            # initial age vector
w_i        = interpolate(Constant(0.0), V)            # initial velocity vector
m          = interpolate(Constant(0.0), V)            # mesh velocity
m_1        = interpolate(Constant(0.0), V)            # prev. mesh velocity

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
bdot      = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
#c         = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2      # Patterson 1994
Ta        = interpolate(Constant(Tavg), V)
c         = cp
k         = 2.1*(rho / rhoi)**2                           # Arthern 2008
Tcoef     = interpolate(Constant(1.0), V)                 # T above Tw = 0.0
Kcoef     = interpolate(Constant(1.0), V)                 # enthalpy coef.
T         = Tcoef * H / c                                 # temperature

# enthalpy residual :
theta     = 0.5
H_mid     = theta*H + (1 - theta)*H_1
f_H       = - k/(rho*c) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
            + (w-m) * H_mid.dx(0) * psi * dx \
            - (H - H_1)/dt * psi * dx


# density residual :
# material derivative :
#  dr   pr     pr
#  -- = -- + w --
#  dt   pt     pz
# SUPG method phihat :        
vnorm     = sqrt(dot(w-m, w-m) + 1e-10)
cellh     = CellSize(mesh)
phihat    = phi + cellh/(2*vnorm)*dot(w-m, phi.dx(0))

theta     = 0.878
rho_mid   = theta*rho + (1 - theta)*rho_1
rhoCoef   = interpolate(Constant(kcHh), V)
drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * (rhoi - rho_mid)
f_rho     = + (rho - rho_1)/dt * phi * dx \
            - drhodt * phihat * dx \
            + (w-m) * rho_mid.dx(0) * phihat * dx 

# velocity residual :
theta     = 0.878
w_mid     = theta*w + (1 - theta)*w_1
# Zwally equation for surface velocity :
f_w       = + rho * w_mid.dx(0) * eta * dx \
            + drhodt * eta * dx
# Arthern equation of strain rate from 'Sorge's Law' :
#f_w       = + rho**2 * w_mid.dx(0) * eta * dx \
#            - bdot * rho.dx(0) * eta * dx

# age residual :
# theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
#               0.5=Crank-Nicolson, 0=Forward-Euler) :
# uses Taylor-Galerkin upwinding :
theta     = 0.5 
a_mid     = theta*a + (1-theta)*a_1
f_a       = + (a - a_1)/dt * xi * dx \
            - 1 * xi * dx \
            + (w-m) * a_mid.dx(0) * xi * dx \
            - 0.5 * ((w-m) - (w_1-m_1)) * a_mid.dx(0) * xi * dx \
            + (w-m)**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
            - (w-m) * (w-m).dx(0) * dt/2 * a_mid.dx(0) * xi * dx

# equation to be minimzed :
f         = f_H + f_rho + f_w
df        = derivative(f, h, dh)   # temp/density jacobian
df_a      = derivative(f_a, a, da) # age jacobian

#===============================================================================
# initialize data structures :

# project the initial functions onto the space and initialize firn object : 
data    = project_vars(V, H, T, rho, drhodt, a, w, k, c, omega)
FEMdata = (mesh, V, MV, H_i, rho_i, w_i, a_i, h, H, T, 
           rho, drhodt, w, m, m_1, a, h_1, a_1, k, c)
firn    = Firn(const, FEMdata, data, bcs, srf_exp, Tavg, rhos, 
               adot, A, z, l, index, dt)

# load initialization data :
#firn.set_ini_conv(ex)

if bp:
  plt.ion() 
  plot = Plot(firn)
  plt.draw()


#===============================================================================
# Compute solution :
tstart = time.clock()
#set_log_active(False)
params = {'newton_solver' : {'relaxation_parameter'    : 1.00,
                             'maximum_iterations'      : 25,
                             'error_on_nonconvergence' : False,
                             'relative_tolerance'      : 1e-10,
                             'absolute_tolerance'      : 1e-10}}
for t in times:
  # update boundary conditions :
  firn.update_Hbc()
  #firn.update_rhoBc()

  # newton's iterative method :
  #h.vector().set_local(h.vector().array() + rand())
  solve(f == 0, h, bcs, J=df, solver_parameters=params)

  # solve for age :
  solve(f_a == 0, a, ageBc, J=df_a, solver_parameters=params)
  
  # adjust the coefficient vectors :
  firn.adjust_vectors(Kcoef, Tcoef, rhoCoef)
  
  # update firn object :
  firn.update_vars(t)
  firn.update_height_history()
  firn.update_height()
  
  # update model parameters :
  if t != times[-1]:
     h_1.assign(h)
     a_1.assign(a)
     m_1.assign(m)

  # update the plotting parameters :
  if bp:
    plot.update_plot()
    plt.draw()

# calculate total time to compute :
tfin = time.clock()
if bp:
  plt.ioff()
  plt.show()

ttot   = tfin - tstart
thours = ttot/60
print "total time to process %i years: %.2e mins" % ((tf - t0)/spy, thours)

# plot the surface height trend :
#plot.plot_height(times, firn.ht, firn.origHt)


