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
rhosi = 300.                   # initial density at surface ..... kg/m^3
rhoin = rhoi                   # initial density ................ kg/m^3
rhow  = 1000.                  # density of water ............... kg/m^3
rhom  = 550.                   # density at 15 m ................ kg/m^3
acc   = 250. / spy             # surface accumulation ........... kg/(m^2 s)
A     = acc/rhosi*1e-3         # surface accumulation ........... mm/s
Va    = 10.                    # mean annual wind speed ......... m/s
cp    = 2009.                  # heat capacity of ice ........... J/(kg K)
ki    = 2.1                    # thermal conductivity of ice .... W/(m K)
dkdz  = 0.0                    # derivative of k with depth ..... W/(m^2 K)
ws    = - acc / rhosi          # velocity of surface snow ....... m/s

# model variables :
n1    = 50                     # num of top z-positions
n2    = 50                     # num of bottom z-positions
omega = 2*pi/spy               # frequency of earth rotations ... rad / s
Ti    = -10.                   # initial temperature ............ degrees C
Tavg  = -10.                   # average temperature ............ degrees C
zs    = 40.                    # surface start .................. m
zm    = zs - 15.               # critical depth ................. m
zb    = 0.                     # depth .......................... m
dz1   = zm/n1                  # initial top z-spacing .......... m
dz2   = ((zs - zm) - zb)/n2    # initial bottom z-spacing ....... m
l1    = dz1*ones(n1+1)         # top height vector .............. m
l2    = dz2*ones(n2+1)         # bottom height vector ........... m
dt    = 0.025*spy              # time-step ...................... s
t0    = 0.0                    # begin time ..................... s
tf    = 50*spy                 # end-time ....................... s

#==============================================================================
# create mesh and define function space :
mesh1  = Interval(n1, zm, zs)   # top mesh
mesh2  = Interval(n2, zb, zm)   # bottom mesh

# refine mesh :
cell_markers = CellFunction("bool", mesh2)
cell_markers.set_all(False)
origin = Point(zm)
for cell in cells(mesh2):
  p  = cell.midpoint()
  if p.distance(origin) < 10:
    cell_markers[cell] = True
mesh2 = refine(mesh2, cell_markers)

# update coordinates :
z1     = mesh1.coordinates()[:,0].copy()            # initial z-coord
z2     = mesh2.coordinates()[:,0].copy()            # initial z-coord
numNew = len(z2) - len(l2)                          # number of split nodes
l2     = l2[:-numNew]                               # remove split heights
l2     = append(l2, dz2/2 * ones(numNew * 2))       # append new split heights
index  = argsort(z2)                                # index of updated mesh

# create function spaces :
V1     = FunctionSpace(mesh1, 'Lagrange', 1)        # function space for rho, T
V2     = FunctionSpace(mesh2, 'Lagrange', 1)        # function space for rho, T
MV1    = V1*V1                                      # mixed function space
MV2    = V2*V2                                      # mixed function space

# cyclical surface temperature :
code  = 'Tavg + 9.9*sin(omega*t)'
Ts1   = Expression(code, Tavg=Tavg, omega=omega, t=0.0)

# midpoint temperature :
Ts2   = Expression('T', T=Ti)

# variable surface density by S.R.M. Ligtenberg et all 2011 :
code  = '-151.94 + 1.4266*(73.6 + 1.06*Tavg + 0.0669*A + 4.77*Va)'
rhoS1 = Expression(code, Tavg=Tavg + 273.15, A=A, Va=Va)

# midpoint density :
rhoS2 = Expression('rho', rho=rhoin)

# define the Dirichletplitsurface boundary :
def surface(x, on_boundary):
  return on_boundary and x[0] == zs

def criticalPoint(x, on_boundary):
  return on_boundary and x[0] == zm

Tbc1 = DirichletBC(MV1.sub(0), Ts1, surface)          # temperature surface
Tbc2 = DirichletBC(MV2.sub(0), Ts2, criticalPoint)    # temperature surface
Dbc1 = DirichletBC(MV1.sub(1), rhoS1, surface)        # density surface
Dbc2 = DirichletBC(MV2.sub(1), rhoS2, criticalPoint)  # density surface

#==============================================================================
# Define variational problem top :
T_i1         = interpolate(Constant(Ti), V1)    # initial temperature vector
rho_i1       = interpolate(Constant(rhoin), V1) # initial density vector
h1           = Function(MV1)                    # solution
h_11         = Function(MV1)                    # previous solution
T_01, rho_01 = split(h_11)                      # initial value functions

dh1          = TrialFunction(MV1)               # trial function for solution
dT1, drho1   = split(dh1)                       # trial functions for T, rho
T1, rho1     = split(h1)                        # solutions for T, rho
j1           = TestFunction(MV1)                # test function in mixed space
psi1, phi1   = split(j1)                        # test functions for T, rho

h_01 = project(as_vector([T_i1,rho_i1]), MV1)  # project inital values on space
h1.vector().set_local(h_01.vector().array())   # initalize T, rho in solution
h_11.vector().set_local(h_01.vector().array()) # initalize T, rho in prev. sol

# expression for variable rate constant KT :
#actEnergy = 1475.0*exp(2.5*T) + 85.4                  # Tyler
#actEnergy = 883.8*abs(T)**(-0.885)
#ko        = 8.36*abs(T)**(-2.061)
#K         = 8.0*ko*exp(-actEnergy/(R*(T + 273.15)))
K1        = 8.36*(T1 + 273.15)**(-2.061)               # Reeh ZL correction

# thermal conductivity Van Dusen formula (lower limit) :
#k = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# thermal conductivity Schwerdtfeger forumla (upper limit) :
#k = (2*ki*rho) / (3*rhoi - rho)

# thermal conductivity Arthern et all 1998 :
k1        = 2.1*(rho1 / rhoi)**2
dkdrho1   = 4.2*rho1/rhoi**2
dkdT1     = 0#derivative(k1, T1, dT1)
dkdz1     = dkdrho1*grad(rho1) + dkdT1*grad(T1)

# expression for vertical velocity of firn :
w1        = - acc / rho1

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_T1      = (rho1*cp*(T1-T_01)*psi1/dt + \
            theta*k1*inner(grad(T1),grad(psi1)) + \
            theta*rho1*cp*w1*grad(T1)*psi1 + \
            theta*dkdz1*grad(T1)*psi1 + \
            (1-theta)*k1*inner(grad(T_01),grad(psi1)) + \
            (1-theta)*rho1*cp*w1*grad(T_01)*psi1+ \
            (1-theta)*dkdz1*grad(T_01)*psi1)*dx

# total derivative d rho / dt :
#drhodt = K*acc*(rhoi - rho)/rhoi                      # Zwally and Li, 2002
#drhodt = (rhoi - rho)*(acc*rhoi/rhow)*8.0*K           # Reeh ZL correction
drhokdt1   = (rhoi - rho1)*(acc*rhoi/rhow)*8.0*K1
drhok_0dt1 = (rhoi - rho_01)*(acc*rhoi/rhow)*8.0*K1

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_rho1    = ((rho1-rho_01)/dt - \
            theta*(drhokdt1 - 4.3*w1*grad(rho1)) - \
            (1-theta)*(drhok_0dt1 - 4.3*w1*grad(rho_01)))*phi1*dx

# equation to be minimzed :
f1        = f_T1 + f_rho1
df1       = derivative(f1, h1, dh1)  # jacobian 

#==============================================================================
# Define variational problem bottom :
T_i2         = interpolate(Constant(Ti), V2)    # initial temperature vector
rho_i2       = interpolate(Constant(rhoin), V2) # initial density vector
h2           = Function(MV2)                    # solution
h_12         = Function(MV2)                    # previous solution
T_02, rho_02 = split(h_12)                      # initial value functions

dh2          = TrialFunction(MV2)               # trial function for solution
dT2, drho2   = split(dh2)                       # trial functions for T, rho
T2, rho2     = split(h2)                        # solutions for T, rho
j2           = TestFunction(MV2)                # test function in mixed space
psi2, phi2   = split(j2)                        # test functions for T, rho

h_02 = project(as_vector([T_i2,rho_i2]), MV2)  # project inital values on space
h2.vector().set_local(h_02.vector().array())   # initalize T, rho in solution
h_12.vector().set_local(h_02.vector().array()) # initalize T, rho in prev. sol

# expression for variable rate constant KT :
#actEnergy = 1475.0*exp(2.5*T) + 85.4                  # Tyler
#actEnergy = 883.8*abs(T)**(-0.885)
#ko        = 8.36*abs(T)**(-2.061)
#K         = 8.0*ko*exp(-actEnergy/(R*(T + 273.15)))
K2        = 8.36*(T2 + 273.15)**(-2.061)               # Reeh ZL correction

# thermal conductivity Van Dusen formula (lower limit) :
#k = 2.1e-2 + 4.2e-4*rho + 2.2e-9*rho**3

# thermal conductivity Schwerdtfeger forumla (upper limit) :
#k = (2*ki*rho) / (3*rhoi - rho)

# thermal conductivity Arthern et all 1998 :
k2        = 2.1*(rho2 / rhoi)**2
dkdrho2   = 4.2*rho2/rhoi**2
dkdT2     = 0#derivative(k2, T2, dT2)
dkdz2     = dkdrho2*grad(rho2) + dkdT2*grad(T2)

# expression for vertical velocity of firn :
w2        = - acc / rho2

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_T2      = (rho2*cp*(T2-T_02)*psi2/dt + \
            theta*k2*inner(grad(T2),grad(psi2)) + \
            theta*rho2*cp*w2*grad(T2)*psi2 + \
            theta*dkdz2*grad(T2)*psi2 + \
            (1-theta)*k2*inner(grad(T_02),grad(psi2)) + \
            (1-theta)*rho2*cp*w2*grad(T_02)*psi2+ \
            (1-theta)*dkdz2*grad(T_02)*psi2)*dx

# total derivative d rho / dt :
#drhodt = K*acc*(rhoi - rho)/rhoi                      # Zwally and Li, 2002
#drhodt = (rhoi - rho)*(acc*rhoi/rhow)*8.0*K           # Reeh ZL correction
drhokdt2   = (rhoi - rho2)*(acc*rhoi/rhow)*8.0*K2
drhok_0dt2 = (rhoi - rho_02)*(acc*rhoi/rhow)*8.0*K2

# theta scheme (1=Backwards-Euler, 0.5=Crank-Nicolson, 0=Forward-Euler) :
theta     = 1.0
f_rho2    = ((rho2-rho_02)/dt - \
            theta*(drhokdt2 - 4.3*w2*grad(rho2)) - \
            (1-theta)*(drhok_0dt2 - 4.3*w2*grad(rho_02)))*phi2*dx

# equation to be minimzed :
f2        = f_T2 + f_rho2
df2       = derivative(f2, h2, dh2)  # jacobian 

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

plt.ion()

fig   = plt.figure(figsize=(13,7))
Tax   = fig.add_subplot(121)
rhoax = fig.add_subplot(122)

Tax.cla()
rhoax.cla()
# format : [xmin, xmax, ymin, ymax]
Tax.axis([Tmin, Tmax, zmin, zmax])
Tax.grid()
rhoax.axis([rhoMin, rhoMax, zmin, zmax])
rhoax.grid()

# top plot :
tplot1   = project(T1, V1).vector().array()
rhoplot1 = project(rho1, V1).vector().array()
phT1,    = Tax.plot(tplot1, z1, 'r-')                         # temp plot
phTs,    = Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)       # temp surface
phTsp1,  = Tax.plot(Th*ones(len(z1)), z1, 'r+')               # height of node
phrho1,  = rhoax.plot(rhoplot1, z1, 'g-')                     # dens plot
phrhoS,  = rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2) # dens surface
phrhoSp1,= rhoax.plot(rhoh*ones(len(z1)), z1, 'r+')           # height of node

#bottom plot :
tplot2   = project(T2, V2).vector().array()
rhoplot2 = project(rho2, V2).vector().array()
phT2,    = Tax.plot(tplot2[index], z2[index], 'r-')           # temp plot
phTsp2,  = Tax.plot(Th*ones(len(z2)), z2[index], 'b+')        # height of node
phrho2,  = rhoax.plot(rhoplot2[index], z2[index], 'g-')       # dens plot
phrhoSp2,= rhoax.plot(rhoh*ones(len(z2)), z2[index], 'b+')    # height of node

# labels and stuff :
fig_text = plt.figtext(.85,.95,'Time = 0.0 yr')
Tax.set_title('Temperature of Firn')
Tax.set_xlabel(r'T $(\degree C)$')
Tax.set_ylabel(r'Depth $(m)$')
rhoax.set_title('Density of Firn')
rhoax.set_xlabel(r'$\rho$ $\left (\frac{kg}{m^3}\right )$')
rhoax.set_ylabel(r'Depth $(m)$')

#==============================================================================
# Compute solution
t = dt
ht = []
while t <= tf:
  # newton's iterative method :
  solve(f1 == 0, h1, [Tbc1, Dbc1], J=df1)
  solve(f2 == 0, h2, [Tbc2, Dbc2], J=df2)
  
  # find vector of T, rho :
  tplot1   = project(T1, V1).vector().array()
  tplot2   = project(T2, V2).vector().array()
  rhoplot1 = project(rho1, V1).vector().array()
  rhoplot2 = project(rho2, V2).vector().array()
  
  # calculate height of each interval (conservation of mass) :
  lnew1    = l1*rhoin / rhoplot1
  lnew2    = l2*rhoin / rhoplot2[index]
  zSum1    = z2[-1] 
  zSum2    = zb
  for i in range(len(z1)):
    z1[i]  = zSum1 + lnew1[i]
    zSum1 += lnew1[i]
  for i in range(len(z2)):
    z2[i]  = zSum2 + lnew2[i]
    zSum2 += lnew2[i]

  # temperature results plot :
  fig_text.set_text('Time = %.2f yr' % (t / spy) )
  phT1.set_xdata(tplot1)
  phT2.set_xdata(tplot2[index])
  phT1.set_ydata(z1)
  phT2.set_ydata(z2)
  
  # density results plot :
  phrho1.set_xdata(rhoplot1)
  phrho2.set_xdata(rhoplot2[index])
  phrho1.set_ydata(z1)
  phrho2.set_ydata(z2)
  
  # surface plots :
  phTs.set_ydata(z1[-1])
  phrhoS.set_ydata(z1[-1])
  
  # heights plots :
  phTsp1.set_ydata(z1)
  phTsp2.set_ydata(z2)
  phrhoSp1.set_ydata(z1)
  phrhoSp2.set_ydata(z2)
  
  plt.draw()
  
  # track the height of the ice :
  if t>35*spy:
    ht.append(z1[-1])
  
  # update boundary conditions, time, and previous solution :
  Ts1.t      = t
  Ts2.T      = tplot1[0]
  rhoS1.Tavg = tplot1[-1] + 273.15
  rhoS2.rho  = rhoplot1[0]
  
  t         += dt
  h_11.assign(h1)
  h_12.assign(h2)

plt.ioff()
plt.show()


