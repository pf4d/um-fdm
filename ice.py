#!/usr/bin/python
from pylab import *
from scipy import sparse
from scipy.integrate import ode
from matplotlib import rc
from math import sin, pi

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
rho   = 911                    # density of ice - kg/m^3
cp    = 2009                   # heat capacity of ice - J/(kg K)
beta  = 9.8e-8                 # pressure dependence of melting point - K/Pa
k     = 2.1                    # thermal diffusivity of ice - W/(m K)
dzsdx = 1.5e-3                 # surface slope of ice
dTdx  = 1.0e-4                 # horizontal temp. gradient - K/m
Qgeo  = 42.0e-3                # geothermal heat-flux - W/m^2

# initial values :
n     = 30                     # num of z-positions
zs    = 0                      # surface start
zb    = -1200                  # depth - m
dt    = 20*spy                 # time-step - s
t0    = 0.0                    # begin time
tf    = 60000*spy              # end-time - s
z     = linspace(0, zb, n)     # z-coordinate corresponding to theta - m
dz    = z[1]                   # vertical step - m
theta = ones(n) * -10          # temperature - degree C
sigma = zeros(n)               # rescaled vertical component
u     = zeros(n)               # horizontal ice velocity - m/a
dudz  = zeros(n)               # partial derivative of u with respect to z
w     = zeros(n)               # vertical ice velocity - m/a
phi   = zeros(n)               # heat sources from deformation of ice - W/m^3
Tpmp  = beta*rho*g*z[-1]       # Pressure melting point of ice at bed - deg. C

# calculate the values for each node :
for i in range(n):
  sigma[i]  = (z[i] - z[-1]) / (z[0] - z[-1])
  u[i]      = 100*sigma[i]**4 / spy  # meters per year
  w[i]      = 0.2*sigma[i] / spy     # meters per year
  if i > 0 :
    dudz[i] = (u[i] - u[i-1]) / (z[i] - z[i-1])
  phi[i]    = rho * g * (z[0] - z[i]) * dudz[i] * dzsdx

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
A = A / dz**2 * k/(rho*cp)  # k/(rho*cp) = m^2/s
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
B[0,0] = 1.0
B[-1,-1] = 1.0
B = B.tocsr()

# solved for u_j (zb upwinded) :
#
# with    dT |
#       k -- |     = Qgeo
#         dz |z=zb 
#
# and     dT   u_j-2 - 4u_j-1 + 3u_j
#         -- = ---------------------
#         dz           2dz
def fix_boundary(y, t):
  if t > 30045 * spy :
    y[0] = 5 + 5*sin(2*pi*t/spy)
  else :
    y[0] = -10 + 5*sin(2*pi*t/spy)             # seasonal var. surface temp
  y[-1] = (-Qgeo*2*dz/k + 4*y[-2] - y[-3])/3.0  # heat flow from bed
  # set max temp :
  for i in range(n):
    if y[i] >= Tpmp :
      y[i] = Tpmp
  return y

# right-hand-side function :
def rhs(t, y):
  y = fix_boundary(y,t)
  return A*y - w*(B * y) - u*dTdx + phi/(rho * cp)

#==============================================================================
# create the ODE Machinery :
i = ode(rhs)
i.set_integrator('vode', method='bdf')
i.set_initial_value(theta,t0)

# animation parameters :
xmax = 0
xmin = -15
ymax = 0
ymin = zb
ion()
clf()
axis([xmin, xmax, ymin, ymax])
grid()
ph,  = plot(theta, z, 'go-')
fig_text = figtext(.70,.75,'Time = 0.0 yr')
#title(r'Heat Distribution in Ice ($\theta$)')
xlabel(r'$\theta$ ($\degree$C)')
ylabel(r'$z$ (m)')

theta_avg = []    # sum of values through time
ts        = []    # time values corresponding to theta_avg
th        = []    # intermediate theta values
th2       = []    # intermediate theta values
zh        = []    # intermediate z values
zh2       = []    # intermediate z values
                  # intermediate time values :
ti = array([700.0, 1500.0, 3000.0, 6500.0, 12500.0, 16000.0, 25000.0])
tk = ti + 30000
ti = ti*spy
tk = tk*spy

# loop to solve linear system for each time step :
while i.t <= tf :
  ts.append(i.t)
  fig_text.set_text('Time = %.0f yr' % (i.t / spy) )
  theta = i.integrate(i.t+dt)
  theta = fix_boundary(i.y, i.t)
  ph.set_xdata(theta)
  draw()
  theta_avg.append( sum(theta) / n )
  # load up intermediate values of theta.
  if any(ti == i.t):
    th.append(theta)
    zh.append(z)
  elif any(tk == i.t):
    th2.append(theta)
    zh2.append(z)
ts = array(ts)/spy; ts = ts.astype('int')
ti = ti/spy; ti = ti.astype('int')
tk = tk/spy; tk = tk.astype('int')
ioff()
show()

plot_ice(th, zh, ti)
plot_ice(th2, zh2, tk)
plot_avg(theta_avg, ts)



