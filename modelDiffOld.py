#!/usr/bin/python
from pylab import *
from scipy import sparse
from scipy.integrate import ode
from scipy.optimize import newton_krylov
from math import sin, pi

#==============================================================================
# plotting function:
def plot_ice(th, zh, ti):
  clf()
  for i in range(len(th)):
    plot(th[i], zh[i], '-', label = "%i yr" % ti[i])
  grid()
  leg = legend()
  ltext  = leg.get_texts()
  frame  = leg.get_frame()
  setp(ltext, fontsize='small')
  frame.set_alpha(0.75)
  frame.set_facecolor('0.80')
  xlabel(r'$\theta$ ($\degree$C)')
  ylabel(r'$z$ (m)')
  show()

# average temperature vs time plot :
def plot_avg(theta_avg, ts):
  clf()
  plot(ts, theta_avg)
  grid()
  xlabel(r'Time (yr)')
  ylabel(r'Average $\theta$ ($\degree$C)')
  show()

#==============================================================================
# constants :
g     = 9.81                   # gravitational acceleration - m/s^2
spy   = 31556926               # seconds per year
rhoi  = 911.                   # density of ice - kg/m^3
rhos  = 300.                   # density of surface snow - kg/m^3
cp    = 2009.                  # heat capacity of ice - J/(kg K)
beta  = 9.8e-8                 # pressure dependence of melting point - K/Pa
ki    = 2.1                    # thermal conductivity of ice - W/(m K)
dzsdx = 1.5e-3                 # surface slope of ice
dTdx  = 1.0e-4                 # horizontal temp. gradient - K/m
Qgeo  = 42.0e-3                # geothermal heat-flux - W/m^2
R     = 8.3144621              # gas constant ................... J/(mol K)
acc   = 250. / spy             # surface accumulation ........... kg/(m^2 a)

# initial values :
n     = 30                     # num of z-positions
zs    = 0.                     # surface start
zb    = -40.                   # depth - m
dt    = 0.025*spy              # time-step - s
t0    = 0.0                    # begin time
tf    = 10*spy                 # end-time - s
z     = linspace(0, zb, n)     # z-coordinate corresponding to theta - m
dz    = -z[1]                  # vertical step - m
theta = ones(n) * -10          # temperature - degree C
rho   = ones(n) * rhoi         # density vector - kg/m^3
rhom1 = rho.copy()             # t-1 density vector - kg/m^3
rhom2 = rho.copy()             # t-2 density vector - kg/m^3  
k     = 2.1*(rho / rhoi)**2    # thermal conductivity vector - W/(m K)
E     = zeros(n)               # activation energy
ko    = zeros(n)
K     = zeros(n)               # rate constant
sigma = zeros(n)               # rescaled vertical component
u     = zeros(n)               # horizontal ice velocity - m/a
dudz  = zeros(n)               # partial derivative of u with respect to z
w     = zeros(n)               # vertical ice velocity - m/a
phi   = zeros(n)               # heat sources from deformation of ice - W/m^3
Tpmp  = beta*rhoi*g*z[-1]      # Pressure melting point of ice at bed - deg. C

# calculate the values for each node :
for i in range(n):
  sigma[i]  = (z[i] - z[-1]) / (z[0] - z[-1])
  u[i]      = 100*sigma[i]**4 / spy  # meters per year
  w[i]      = 0.2*sigma[i] / spy     # meters per year
  if i > 0 :
    dudz[i] = (u[i] - u[i-1]) / (z[i] - z[i-1])
  phi[i]    = rho[i] * g * (z[0] - z[i]) * dudz[i] * dzsdx

w     = - acc / rho
#==============================================================================
# diffusion matrix :
#
# using   d^2T   u_j-1 - 2u_j + u_j+1
#         ---- = --------------------
#         dz^2          dz^2
A = sparse.lil_matrix((n, n))
A.setdiag(-2*ones(n))
A.setdiag(ones(n-1), k=1)
A.setdiag(ones(n-1), k=-1)
A[0,:] = zeros(n)
A[-1,:] = zeros(n)
A = A / dz**2
A = A.tocsr()

# vertical ice advection (upwinded) :
#
# using   dT   -3u_j + 4u_j+1 - u_j+2
#         -- = ----------------------
#         dz           2dz
B = sparse.lil_matrix((n, n))
B.setdiag(-3*ones(n))
B.setdiag(4*ones(n-1), k=1)
B.setdiag(-1*ones(n-1), k=2)
B = B/(2*dz)
B[0,:] = zeros(n)
B[-1,:] = zeros(n)
#B[0,0] = 1.0
B[1,1] = 1.0
B = B.tocsr()

#==============================================================================
# solver functions :

# calculate variables :
def calc_variables(y, rhom2, rhom1, rho, k, w):
  # rate constant :
  E      = 883.8*abs(y)**(-0.885)
  ko     = 8.36*abs(y)**(-2.061)
  K      = 8.0*ko*exp(-E/(R*(y + 273.15)))
  # velocity :
  w      = - acc / rho
  # density :
  rho    = rhom1 + dt*( K*acc*(rhoi - rho)/rhoi - w*(B * rho) )
  #rho    = 4*rhom1 - rhom2 + \
  #         (2*dt/3)*( K*acc*(rhoi - rho)/rhoi - w*(B * rho) )
  rho[0] = rhos    # update density surface condition
  #rho[-1] = rhoi
  # thermal conductivity :
  k      = 2.1*(rho / rhoi)**2
  # return the variables please :
  return rho, k, w

# update the boundary conditions :
def fix_boundary(y, t):
  # surface temp change :
  y[0] = -10 + 9.9*sin(2*pi*t/spy)           # seasonal var. surface temp
  # set max temp :
  for i in range(n):
    if y[i] >= Tpmp :
      y[i] = Tpmp
  return y


# right-hand-side function :
def rhs(t, y, rhom2, rhom1, rho, k, w):
  rho, k, w = calc_variables(y, rhom2, rhom1, rho, k, w)
  y = fix_boundary(y,t)
  def residual(y):
    return - y + k/(rho * cp)*(A * y) - w*(B * y) - u*dTdx + phi/(rho * cp)
  #y = newton_krylov(residual, y, method='lgmres', verbose=0)
  return k/(rho * cp)*(A * y) - w*(B * y) - u*dTdx + phi/(rho * cp)
  #return y

#==============================================================================
# create the ODE Machinery :
i = ode(rhs)
i.set_integrator('vode', method='bdf')
i.set_initial_value(theta, t0)
i.set_f_params(rhom2, rhom1, rho, k, w)

#==============================================================================
# animation parameters :
Tmin   = -20                                 # T x-coord min
Tmax   = 0                                   # T x-coord max
Th     = Tmin + 0.1*(Tmax - Tmin) / 2        # T height x-coord
rhoMin = 250                                 # rho x-coord min
rhoMax = 1000                                # rho x-coord max
rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord
zmax   = zs + 15                             # max z-coord
zmin   = zb                                  # min z-coord

ion()

fig   = figure(figsize=(13,7))
Tax   = fig.add_subplot(121)
rhoax = fig.add_subplot(122)

Tax.cla()
rhoax.cla()
# format : [xmin, xmax, ymin, ymax]
Tax.axis([Tmin, Tmax, zmin, zmax])
Tax.grid()
rhoax.axis([rhoMin, rhoMax, zmin, zmax])
rhoax.grid()

phT,    = Tax.plot(theta, z, 'r-')                           # temp plot
phTs,   = Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)       # temp surface
phTsp,  = Tax.plot(Th*ones(n), z, 'r+')                      # height of node
phrho,  = rhoax.plot(rho, z, 'g-')                           # dens plot
phrhoS, = rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2) # dens surface
phrhoSp,= rhoax.plot(rhoh*ones(n), z, 'r+')                  # height of node

fig_text = figtext(.85,.95,'Time = 0.0 yr')
Tax.set_title('Temperature of Firn')
Tax.set_xlabel(r'T $(\degree C)$')
Tax.set_ylabel(r'Depth $(m)$')
rhoax.set_title('Density of Firn')
rhoax.set_xlabel(r'$\rho$ $\left (\frac{kg}{m^3}\right )$')
rhoax.set_ylabel(r'Depth $(m)$')

theta_avg = []    # sum of values through time
ts        = []    # time values corresponding to theta_avg

#==============================================================================
# loop to solve linear system for each time step :
while i.t <= tf :
  ts.append(i.t)
  fig_text.set_text('Time = %.2f yr' % (i.t / spy) )
  theta = i.integrate(i.t+dt)
  theta = fix_boundary(i.y, i.t)
  rhom2 = rhom1.copy()
  rhom1 = rho.copy()
  rho, k, w = calc_variables(theta, *i.f_params)
  i.set_f_params(rhom2, rhom1, rho, k, w)
  
  phT.set_xdata(theta)
  phrho.set_xdata(rho)
  draw()
  theta_avg.append( sum(theta) / n )

ioff()
show()

#plot_avg(theta_avg, ts)



