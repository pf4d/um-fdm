"""
model.py
Evan Cummings
05.23.12

FEniCS solution to firn temperature profile.

"""

from numpy import *
from dolfin import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from scipy.interpolate import interp1d

class FixedOrderFormatter(ScalarFormatter):
  """Formats axis ticks using scientific notation with a constant order of 
  magnitude"""
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                             useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
    self.orderOfMagnitude = self._order_of_mag

#==============================================================================
# constants :
pi    = 3.141592653589793      # pi
g     = 9.81                   # gravitational acceleration ..... m/s^2
R     = 8.3144621              # gas constant ................... J/(mol K)
spy   = 31556926.0             # seconds per year ............... s/a
rhoi  = 917.                   # density of ice ................. kg/m^3
rhosi = 300.                   # initial density at surface ..... kg/m^3
rhoin = rhoi                   # initial density ................ kg/m^3
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
beta  = 1.5                    # drhodt smoothing
omega = 2*pi/spy               # frequency of earth rotations ... rad / s
Tavg  = Tw - 10.0              # average temperature ............ degrees K
zs    = 40.                    # surface start .................. m
zb    = 0.                     # depth .......................... m
dz    = (zs - zb)/n            # initial z-spacing .............. m
l     = dz*ones(n+1)           # height vector .................. m
dt    = 0.025*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = 220*spy                # end-time ....................... s

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
#code  = '300 + 700*( (Ts - Tmin) / (Tmax - Tmin) )'   # evan experiment
#rhoS  = Expression(code, Tmax=Tavg + 9.9, Tmin=Tavg - 9.9, Ts=Tavg)
#rhoS  = Constant(300)

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
rho_i      = interpolate(Constant(rhoin), V) # initial density vector
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
#actEnergy = 1475.0*exp(2.5*T) + 85.4                  # Tyler
#actEnergy = 883.8*T**(-0.885)
#ko        = 8.36*T**(-2.061)
#K         = 8.0*ko*exp(-actEnergy/(R*T))
K         = 8.36*T**(-2.061)                          # Reeh ZL correction
dK        = 8.36*-2.061*T**(-1.061)                   # derivative K(T)

# thermal conductivity Van Dusen formula (lower limit) :
#k = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# thermal conductivity Schwerdtfeger forumla (upper limit) :
#k = (2*ki*rho) / (3*rhoi - rho)

# thermal conductivity Arthern et all 1998 :
k         = 2.1*(rho / rhoi)**2
dkdrho    = 4.2*(rho / rhoi**2)
#dkdT      = 9.828*-5.7e-3*exp(-5.7e-3 * T)    # Patterson pg. 205
drhodT    = ( (dK*acc)*(-K*acc/rhoi - 1/dt) - 
              (w*grad(rho) - rho_0/dt - K*acc)*(-dK*acc/rhoi) ) / \
            (-K*acc/rhoi - 1/dt)**2
dkdT      = 4.2*(drhodT / rhoi**2)
dkdz      = dkdrho*grad(rho) + dkdT*grad(T)

km1       = 2.1*(rho_0 / rhoi)**2
dkdrhom1  = 4.2*(rho_0 / rhoi**2)
#dkdTm1    = 9.828*-5.7e-3*exp(-5.7e-3 * T_0)  # Patterson pg. 205
dkdTm1    = ( (dK*acc)*(-K*acc/rhoi - 1/dt) - 
              (w*grad(rho) - rho_0/dt - K*acc)*(-K*acc/rhoi) ) / \
            (-K*acc/rhoi - 1/dt)**2
dkdzm1    = dkdrho*grad(rho_0) + dkdT*grad(T_0)

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_T       = (rho*cp*(T-T_0)*psi/dt + \
            theta*k*inner(grad(T),grad(psi)))*dx # + \
#            theta*rho*cp*w*grad(T)*psi + \
#            theta*dkdz*grad(T)*psi + \
#            (1-theta)*km1*inner(grad(T_0),grad(psi)) + \
#            (1-theta)*rho_0*cp*wm1*grad(T_0)*psi+ \
#            (1-theta)*dkdzm1*grad(T_0)*psi)*dx

# total derivative d rho / dt :
#drhodt = K*acc*(rhoi - rho)/rhoi                      # Zwally and Li, 2002
#drhodt   = (rhoi - rho)*(acc*rhoi/rhow)*beta*K        # Reeh ZL correction
#drho_0dt = (rhoi - rho_0)*(acc*rhoi/rhow)*beta*K      # Reeh ZL correction

# total derivative drhodt from Arthern 2010 :
rhoCoef  = interpolate(Constant(kcHh), V)
drhodt   = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T) + Eg/(R*Tavg) )*(rhoi - rho)
drho_0dt = (acc*g*rhoCoef/kg)*exp( -Ec/(R*T_0) + Eg/(R*Tavg) )*(rhoi - rho_0)

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_rho     = ((rho-rho_0)/dt - \
            theta*(drhodt - w*grad(rho)) - \
            (1-theta)*(drho_0dt - wm1*grad(rho_0)))*phi*dx

# equation to be minimzed :
f         = f_T + f_rho
df        = derivative(f, h, dh)  # jacobian 

#==============================================================================
# animation parameters :
Tmin   = -20                                 # T x-coord min
Tmax   = 0                                   # T x-coord max
Th     = Tmin + 0.1*(Tmax - Tmin) / 2        # T height x-coord

rhoMin = 300                                 # rho x-coord min
rhoMax = 1000                                # rho x-coord max
rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord

wMin   = -22.0e-6
wMax   = -7.5e-6
wh     = wMin + 0.1*(wMax - wMin) / 2

kMin   = 0.0
kMax   = 2.2
kh     = kMin + 0.1*(kMax - kMin) / 2

zmax   = zs + 20                             # max z-coord
zmin   = zb                                  # min z-coord

plt.ion()

fig   = plt.figure(figsize=(16,6))
Tax   = fig.add_subplot(141)
rhoax = fig.add_subplot(142)
wax   = fig.add_subplot(143)
kax   = fig.add_subplot(144)

# format : [xmin, xmax, ymin, ymax]
Tax.axis([Tmin, Tmax, zmin, zmax])
Tax.grid()
rhoax.axis([rhoMin, rhoMax, zmin, zmax])
rhoax.grid()
rhoax.xaxis.set_major_formatter(FixedOrderFormatter(2))
wax.axis([wMin, wMax, zmin, zmax])
wax.grid()
wax.xaxis.set_major_formatter(FixedOrderFormatter(-6))
kax.axis([kMin, kMax, zmin, zmax])
kax.grid()

# x-values :
tplot    = project(T, V).vector().array()[index]
rhoplot  = project(rho, V).vector().array()[index]
wplot    = - spy * acc / rhoplot
kplot1   = 2.1e-2 + 4.2e-4*rhoplot + 2.2e-9*rhoplot**3 # Van Dusen (lower limit)
kplot2   = 2.1*(rhoplot / rhoi)**2             # Arthern et all 1998
kplot3   = (2*ki*rhoplot) / (3*rhoi - rhoplot) # Schwerdtfeger (upper limit)

# y-value :
z = z[index]

# original surface height :
origZ    = zs

# plots :
phT,     = Tax.plot(tplot - 273.15, z, 'r-')                  # temp plot
phTs,    = Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)       # surface
phTs_0,  = Tax.plot(Th, origZ, 'ko')                          # original surface
phTsp,   = Tax.plot(Th*ones(len(z)), z, 'r+')                 # height of nodes

phrho,   = rhoax.plot(rhoplot, z, 'g-')                       # dens plot
phrhoS,  = rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2) # surface
phrhoS_0,= rhoax.plot(rhoh, origZ, 'ko')                      # orininal surface
phrhoSp, = rhoax.plot(rhoh*ones(len(z)), z, 'r+')             # height of nodes

phw,     = wax.plot(wplot, z, 'b-')                           # velocity plot
phws,    = wax.plot([wMin, wMax], [zs, zs], 'k-', lw=2)       # surface
phws_0,  = wax.plot(wh, origZ, 'ko')                          # orininal surface
phwsp,   = wax.plot(wh*ones(len(z)), z, 'r+')                 # height of nodes

phk1,    = kax.plot(kplot1, z, label='Van Dusen')             # k1 plot
phk2,    = kax.plot(kplot2, z, label='Arthern')               # k2 plot
phk3,    = kax.plot(kplot3, z, label='Schwerdtfeger' )        # k3 plot
phks,    = kax.plot([kMin, kMax], [zs, zs], 'k-', lw=2)       # surface
phks_0,  = kax.plot(kh, origZ, 'ko')                          # orininal surface
phksp,   = kax.plot(kh*ones(len(z)), z, 'r+')                 # height of nodes

# formatting :
fig_text = plt.figtext(.85,.95,'Time = 0.0 yr')

Tax.set_title('Temperature')
Tax.set_xlabel(r'$T$ $[\degree C]$')
Tax.set_ylabel(r'Depth $[m]$')

rhoax.set_title('Density')
rhoax.set_xlabel(r'$\rho$ $\left [\frac{kg}{m^3}\right ]$')
#rhoax.set_ylabel(r'Depth $[m]$')

wax.set_title('Velocity')
wax.set_xlabel(r'$w$ $\left [\frac{mm}{s}\right ]$')
#wax.set_ylabel(r'Depth $[m]$')

kax.set_title('Thermal Conductivity')
kax.set_xlabel(r'$k$ $\left [\frac{J}{m K s} \right ]$')
#kax.set_ylabel(r'Depth $[m]$')
# Legend formatting:
leg = kax.legend(loc='lower center')
ltext  = leg.get_texts()
frame  = leg.get_frame()
plt.setp(ltext, fontsize='small')
frame.set_alpha(0)

#==============================================================================
# Compute solution
t      = dt
ht     = []
origHt = []
while t <= tf:
  # newton's iterative method :
  solve(f == 0, h, [Tbc, Dbc], J=df)
  
  # find vector of T, rho :
  tplot   = project(T, V).vector().array()
  rhoplot = project(rho, V).vector().array()
 
  # calculate other data :
  wplot   = -acc / rhoplot * 1e3
  kplot1  = 2.1e-2 + 4.2e-4*rhoplot + 2.2e-9*rhoplot**3
  kplot2  = 2.1*(rhoplot / rhoi)**2
  kplot3  = (2*ki*rhoplot) / (3*rhoi - rhoplot)
  
  # calculate height of each interval (conservation of mass) :
  lnew    = l*rhoin / rhoplot[index]
  zSum    = zb
  for i in range(len(z)):
    z[i]  = zSum + lnew[i]
    zSum += lnew[i]
  
  # correct original height with initial surface conditions :
  if t == dt:
    origZ = z[-1]

  # update plots :
  fig_text.set_text('Time = %.2f yr' % (t / spy) )
  phT.set_xdata(tplot[index] - 273.15)
  phT.set_ydata(z)
  phTs.set_ydata(z[-1])
  phTs_0.set_ydata(origZ)
  phTsp.set_ydata(z)
  
  phrho.set_xdata(rhoplot[index])
  phrho.set_ydata(z)
  phrhoS.set_ydata(z[-1])
  phrhoS_0.set_ydata(origZ)
  phrhoSp.set_ydata(z)
 
  phw.set_xdata(wplot[index])
  phw.set_ydata(z)
  phws.set_ydata(z[-1])
  phws_0.set_ydata(origZ)
  phwsp.set_ydata(z)
  
  phk1.set_xdata(kplot1[index])
  phk2.set_xdata(kplot2[index])
  phk3.set_xdata(kplot3[index])
  phk1.set_ydata(z)
  phk2.set_ydata(z)
  phk3.set_ydata(z)
  phks.set_ydata(z[-1])
  phks_0.set_ydata(origZ)
  phksp.set_ydata(z)
  
  plt.draw()
  
  # calculate the new height of original surface
  # by interpolating vertical speed from w :
  if origZ > z[0]:
    interp = interp1d(z, wplot[index])
    zint   = array([origZ])
    wOrigZ = interp(array([origZ]))
    origZ += wOrigZ[0] / 1e3 * dt
  else:
    origZ  = 0.0

  # track the current height and original surface height of the firn :
  ht.append(z[-1])
  origHt.append(origZ)

  # update kc term in drhodt :
  # if rho >  550, kc = kcHigh
  # if rho <= 550, kc = kcLow
  rhoCoefNew             = ones(len(rhoplot))
  rhoHigh                = where(rhoplot > 550)
  rhoLow                 = where(rhoplot <= 550)
  rhoCoefNew[rhoHigh[0]] = kcHh
  rhoCoefNew[rhoLow[0]]  = kcLw
  rhoCoef.vector().set_local(rhoCoefNew)

  # update boundary conditions, time, and previous solution :
  Ts.t      = t
  rhoS.Ts   = tplot[index][-1]
  #rhoS.Ts   = tplot[index][-1]
  t        += dt
  h_1.assign(h)
  
plt.ioff()
plt.show()

# plot the surface height information :
x = linspace(0, t/spy, len(ht))
plt.plot(x, ht,     label='Surface Height')
plt.plot(x, origHt, label='Original Surface')
plt.xlabel(r'time $[a]$')
plt.ylabel(r'height $[m]$')
plt.title('Surface Height Changes')
plt.grid()
# Legend formatting:
leg = plt.legend(loc='upperr right')
ltext  = leg.get_texts()
frame  = leg.get_frame()
plt.setp(ltext, fontsize='small')
frame.set_alpha(0)
plt.show()



