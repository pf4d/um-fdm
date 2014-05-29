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
from physics   import *
from physical_constants import *
import sys
import time


#===============================================================================
# constants :

# model variables :
spy   = 31556926.0             # seconds per year ............... s/a
cpi   = 2009.                  # const. heat capacitity of ice .. J/(kg K)
Tw    = 273.15                 # triple point water ............. degrees K
n     = 100                    # num of z-positions
rhos  = 360.                   # initial density at surface ..... kg/m^3
rhoi  = 917.                   # density of ice ................. kg/m^3
rhoin = rhoi                   # initial density at surface ..... kg/m^3
adot  = 0.25                   # accumulation rate .............. m/a
Tavg  = Tw - 10.0              # average temperature ............ degrees K

cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
cp    = cpi                    # heat capacity of ice ........... J/(kg K)
zs    = 0.                     # surface start .................. m
zb    = -100.                  # depth .......................... m
dt    = 10.0*spy               # time-step ...................... s
dt    = 0.025*spy               # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s
numt  = (tf-t0)/dt             # number of time steps ........... none
times = linspace(dt,tf,numt)   # array of times to evaluate ..... s
bp    = int(sys.argv[2])       # plot or not .................... bool

# enthalpy surface condition with cyclical 2-meter air temperature :
code    = 'c*( Tavg + 10.0*(sin(2*omega*t) + 5*sin(4*omega*t)))'
H_exp   = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0)

# experimental surface density :
#code    = 'dp*rhon + (1 - dp)*rhoi'
#rho_exp = Expression(code, rhon=rhos, rhoi=rhoi, dp=1e-3)

# constant surface density :
rho_exp = Expression('rhon', rhon=rhos)

# velocity of surface (-acc / rhos) [m/s] :
code    = '- rhoi/rhos * adot / spy'
w_exp   = Expression(code, rhoi=rhoi, adot=adot, spy=spy, rhos=rhos)

firn = Firn(Tavg, rhoin, rhos, adot, dt)
firn.set_geometry(zs, zb)
firn.generate_uniform_mesh(n)
firn.set_parameters(FirnParameters())
firn.set_boundary_conditions(H_exp, rho_exp, w_exp)
firn.initialize_variables()

# load initialization data :
#firn.set_ini_conv(ex)

if bp:
  plt.ion() 
  plot = Plot(firn)
  plt.draw()


#===============================================================================
# form the physics :
fe = Enthalpy(firn)
fa = Age(firn)


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
  fe.solve(params)

  # solve for age :
  fa.solve(params)
  
  # adjust the coefficient vectors :
  firn.adjust_vectors()
  
  # update firn object :
  firn.update_vars(t)
  firn.update_height_history()
  firn.update_height()
  
  # update model parameters :
  if t != times[-1]:
     firn.h_1.assign(firn.h)
     firn.a_1.assign(firn.a)
     firn.m_1.assign(firn.m)

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


