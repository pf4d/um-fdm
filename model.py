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

FEniCS solution to firn enthalpy / density profile.  First arg should is the
end time in years, second is boolean val to plot the solution or not.

"""

from numpy              import *
from fenics             import *
from plot               import *
from firn               import *
from solvers            import TransientSolver
from physical_constants import *
from scipy.interpolate  import interp1d
from scipy.io           import loadmat
import sys
import time


#===============================================================================
# constants :

# model variables :
spy   = 365*24*60*60           # seconds per year ............... s/a
cpi   = 2009.                  # const. heat capacitity of ice .. J/(kg K)
Tw    = 273.15                 # triple point water ............. degrees K
n     = 100                    # num of z-positions
rhos  = 360.                   # initial density at surface ..... kg/m^3
rhoi  = 917.                   # density of ice ................. kg/m^3
rhoin = 917.                   # initial density at surface ..... kg/m^3
rin   = 0.0005**2              # initial grain radius ...........m^2
adot  = 0.1                    # accumulation rate .............. m/a
Tavg  = Tw - 20.0              # average temperature ............ degrees K
Tin   = Tavg
adoti = adot

cp    = 152.5 + 7.122*Tavg     # heat capacity of ice ........... J/(kg K)
cp    = cpi                    # heat capacity of ice ........... J/(kg K)
zs    = 0.                     # surface start .................. m
zb    = -10.0                  # depth .......................... m
dt    = 0.5/365.0*spy          # time-step ...................... s
#dt    = 10.0*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = sys.argv[1]            # end-time ....................... string
tf    = float(tf)*spy          # end-time ....................... s
bp    = int(sys.argv[2])       # plot or not .................... bool
tm    = 0.0
  
# enthalpy BC :
code    = 'c*( Tavg + 5.0*(sin(2*omega*t) + 5*sin(4*omega*t)))'
H_exp   = Expression(code, c=cp, Tavg=Tavg, omega=pi/spy, t=t0)

# surface density :
rho_exp = Expression('rhon', rhon=rhos)
#rho_exp = Constant(rhos)

# velocity of surface (-acc / rhos) [m/s] :
code    = '- rhoi/rhos * adot / spy'
w_exp   = Expression(code, rhoi=rhoi, adot=adot, spy=spy, rhos=rhos)

# grain radius of surface [cm^2] :
r_exp   = Expression('r_s', r_s=rin)


#===============================================================================
# initialize the firn object :
firn = Firn(Tin, rhoin, rin, rhos, adoti, dt)
firn.set_geometry(zs, zb)
firn.generate_uniform_mesh(n)
firn.refine_mesh(divs=3, i=1/3., k=1/20.)
firn.refine_mesh(divs=2, i=1/5., k=1/4.)
#firn.refine_mesh(divs=2, i=1/5., k=1/4.)
firn.calculate_boundaries()
firn.set_parameters(FirnParameters())
firn.set_boundary_conditions(H_exp, rho_exp, w_exp, r_exp)
firn.initialize_variables()

# load initialization data :
#firn.set_ini_conv(ex)

#set_log_active(False)
params = {'newton_solver' : {'relaxation_parameter'    : 1.00,
                             'maximum_iterations'      : 25,
                             'error_on_nonconvergence' : False,
                             'relative_tolerance'      : 1e-10,
                             'absolute_tolerance'      : 1e-10}}

config = { 'mode'                  : 'transient',
           't_start'               : t0,
           't_mid'                 : tm,
           't_end'                 : tf,
           'time_step'             : dt,
           'dt_list'               : None,
           'output_path'           : '.',
           'log'                   : True,
           'coupled' : 
           { 
             'on'                  : False,
             'inner_tol'           : 0.0,
             'max_iter'            : 0
           },
           'enthalpy' : 
           { 
             'on'                  : False,
             'use_surface_climate' : False,
             'T_surface'           : None,
             'q_geo'               : None,
             'lateral_boundaries'  : None,
             'solver_params'       : params,
             'log'                 : True 
           },
           'free_surface' :
           { 
             'on'                  : False
           },  
           'age' : 
           { 
             'on'                  : True,
             'use_smb_for_ela'     : True,
             'ela'                 : None,
             'solver_params'       : params,
           },
           'surface_climate' : 
           { 
             'on'                  : False,
             'T_ma'                : None,
             'T_ju'                : None,
             'beta_w'              : None,
             'sigma'               : None,
             'precip'              : None
           },
           'plot' :
           {
             'on'                  : bp,
             'zMin'                : -1.5, 
             'zMax'                : 0.3,
             'wMax'                : 5,
             'wMin'                : -30,
             'rhoMax'              : 1000,
             'ageMax'              : 100, 
             'omegaMax'            : 0.10, 
           }}

F = TransientSolver(firn, config)

tstart = time.clock()
F.solve()
tfin = time.clock()

ttot   = tfin - tstart
thours = ttot/60
print "total time to process %i years: %.2e mins" % ((tf - t0)/spy, thours)

# plot the surface height trend :
#F.plot.plot_height(F.times, firn.ht, firn.origHt)



