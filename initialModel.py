"""
model.py
Evan Cummings
05.23.12

FEniCS solution to firn temperature profile.

"""

from numpy import *
from dolfin import *
import matplotlib.pyplot as plt

#==============================================================================
# constants :
pi    = 3.141592653589793      # pi
g     = 9.81                   # gravitational acceleration ..... m/s^2
R     = 8.3144621              # gas constant ................... J/(mol K)
spy   = 31556926.0             # seconds per year ............... s/a
rhoi  = 917.                   # density of ice ................. kg/m^3
rhosi = 332.                   # initial density at surface ..... kg/m^3
rhoin = rhoi                   # initial density of ice ......... kg/m^3
rhow  = 1000.                  # density of water ............... kg/m^3
rhom  = 550.                   # density at 15 m ................ kg/m^3
acc   = 250. / spy             # surface accumulation ........... kg/(m^2 s)
A     = acc/rhosi*1e-3         # surface accumulation ........... mm/s
cp    = 2009.                  # heat capacity of ice ........... J/(kg K)
ki    = 2.1                    # thermal conductivity of ice .... W/(m K)
dkdz  = 0.0                    # derivative of k with depth ..... W/(m^2 K)
ws    = - acc / rhosi          # velocity of surface snow ....... m/s

# model variables :
n     = 50                     # num of z-positions
omega = 2*pi/spy               # frequency of earth rotations ... rad / s
Ti    = -10.                   # initial temperature ............ degrees C
Tavg  = Ti                     # average temperature ............ degrees C
zs    = 40.                    # surface start .................. m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.025*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = 500.*spy               # end-time ....................... s

#==============================================================================
# create mesh and define function space :
mesh  = Interval(n, zb, zs)
z     = mesh.coordinates()[:,0]              # initial z-coords
V     = FunctionSpace(mesh, 'Lagrange', 1)   # function space for rho, T
MV    = V*V                                  # mixed function space

# cyclical surface temperature :
code  = 'Tavg + 9.9*sin(omega*t)'
Ts    = Expression(code, Tavg=Tavg, omega=omega, t=0.0)

# variable surface density by S.R.M. Ligtenberg et all 2011 :
code  = '-151.94 + 1.4266*(73.6 + 1.06*Ts + 0.0669*A + 4.77*Va)'
rhoS  = Expression(code, Ts=Ti + 273.15, A=A, Va=10)

# define the Dirichlet surface boundary :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

Tbc = DirichletBC(MV.sub(0), Ts, surface)    # temperature surface
Dbc = DirichletBC(MV.sub(1), rhoS, surface)  # density surface

#==============================================================================
# Define variational problem :
T_i        = interpolate(Constant(Ti), V)    # initial temperature vector
rho_i      = interpolate(Constant(rhoin), V) # initial density vector
h          = Function(MV)                    # solution
h_1        = Function(MV)                    # previous solution
T_0, rho_0 = split(h_1)                      # initial value functions

dh         = TrialFunction(MV)               # trial function for solution
dT, drho   = split(dh)                       # trial functions for T, rho
T,rho      = split(h)                        # solutions for T, rho
j          = TestFunction(MV)                # test function in mixed space
psi, phi   = split(j)                        # test functions for T, rho

h_0 = project(as_vector([T_i,rho_i]), MV)    # project inital values on space
h.vector().set_local(h_0.vector().array())   # initalize T, rho in solution
h_1.vector().set_local(h_0.vector().array()) # initalize T, rho in prev. sol

# expression for variable rate constant KT :

#Equations Tyler found by fitting 

#Activation Energy
#actEnergy = 1475.0*exp(2.5*T) + 85.4        # Tyler
actEnergy = 883.8*abs(T)**(-0.885)
ko        = 8.36*abs(T)**(-2.061)
K         = 8.*ko*exp(-actEnergy/(R*(T + 273.15)))
#K         = 8.36*(T + 273.15)**(-2.061)               # Reeh ZL correction

# Van Dusen formula (lower limit) :
k = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# Schwerdtfeger forumla (upper limit) :
#k = (2*ki*rho) / (3*rhoi - rho)

# Arthern et all 1998 :
#k = 2.1*(rho / rhoi)**2

# expression for vertical velocity of firn :
w = - acc / rho

# residuals
f_T = (rho*cp*(T-T_0)*psi/dt + \
      k*inner(grad(T),grad(psi)) + \
      rho*cp*w*psi*grad(T))*dx

# total derivative d rho / dt :
drhodt = K*acc*(rhoi - rho)/rhoi                   	# Zwally and Li, 2002
#drhodt = (rhoi - rho)*(acc*rhoi/rhow)*8.0*K          # Reeh Zwally Li correction
#drhodt = .03*g*acc*(rhoi - rho)*exp(-actEnergy/(R*(T + 273.15)))                     # Ligtenberg et al.


# d rho/dt material derivative residual :
f_rho = ((rho-rho_0)/dt + 4.5*w*rho.dx(0) - drhodt)*phi*dx

# equation to be minimzed :
f = f_T + f_rho
df = derivative(f, h, dh)  # jacobian 

#==============================================================================
# animation parameters :
Tmin   = -20                                 # T x-coord min
Tmax   = 0                                   # T x-coord max
Th     = Tmin + 0.1*(Tmax - Tmin) / 2        # T height x-coord
rhoMin = 250                                 # rho x-coord min
rhoMax = 1000                                # rho x-coord max
rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord
zmax   = zs + 10                             # max z-coord
zmin   = zb                                  # min z-coord

plt.ion()

fig   = plt.figure(figsize=(14,7))
Tax   = fig.add_subplot(121)
rhoax = fig.add_subplot(122)

Tax.cla()
rhoax.cla()
# format : [xmin, xmax, ymin, ymax]
Tax.axis([Tmin, Tmax, zmin, zmax])
Tax.grid()
rhoax.axis([rhoMin, rhoMax, zmin, zmax])
rhoax.grid()

tplot   = project(T, V)
rhoplot = project(rho, V)
phT,    = Tax.plot(tplot.vector().array(), z, 'r-')           # temp plot
phTs,   = Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)        # temp surface
phTsp,  = Tax.plot(Th*ones(n+1), z, 'r+')                     # height of node
phrho,  = rhoax.plot(rhoplot.vector().array(), z, 'g-')       # dens plot
phrhoS, = rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2)  # dens surface
phrhoSp,= rhoax.plot(rhoh*ones(n+1), z, 'r+')                 # height of node

fig_text = plt.figtext(.85,.95,'Time = 0.0 yr')
Tax.set_title('Temperature of Firn')
Tax.set_xlabel(r'T $(\degree C)$')
Tax.set_ylabel(r'z $(m)$')
rhoax.set_title('Density of Firn')
rhoax.set_xlabel(r'$\rho$ $\left (\frac{kg}{m^3}\right )$')
rhoax.set_ylabel(r'z $(m)$')

#==============================================================================
# Compute solution
t = dt
ht = []
while t <= tf:
  # newton's iterative method :
  solve(f == 0, h, [Tbc, Dbc], J=df)
  
  # find vector of T, rho :
  tplot = project(T, V)
  tplot = tplot.vector().array()
  rhoplot = project(rho, V)
  rhoplot = rhoplot.vector().array()
  
  # calculate height of each interval (conservation of mass athern_2010) :
  lnew = l*rhoin / rhoplot
  zSum = zb
  for i in range(n+1):
    z[i] = zSum + lnew[i]
    zSum += lnew[i]
  
  # update plot :
  fig_text.set_text('Time = %.2f yr' % (t / spy) )
  phT.set_xdata(tplot)
  phT.set_ydata(z)
  phTs.set_ydata(z[-1])
  phTsp.set_ydata(z)
  
  phrho.set_xdata(rhoplot)
  phrho.set_ydata(z)
  phrhoS.set_ydata(z[-1])
  phrhoSp.set_ydata(z)
  plt.draw()
  
  # track the height of the ice :
  if t>35*spy:
    ht.append(z[-1])
  
  # update boundary conditions, time, and previous solution :
  Ts.t = t
  rhoS.Ts = tplot[-1] + 273.15
  t += dt
  h_1.assign(h)
  
plt.ioff()
plt.show()


