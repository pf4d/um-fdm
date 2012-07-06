"""
FEniCS tutorial demo program: Diffusion equation with Dirichlet
conditions and a solution that will be exact at all nodes.
"""

from dolfin import *
from numpy import *

#==============================================================================
# constants :
g     = 9.81                   # gravitational acceleration - m/s^2
spy   = 31556926               # seconds per year
rho   = 911                    # density of ice - kg/m^3
cp    = 2009                   # heat capacity of ice - J/(kg K)
k     = 2.1                    # thermal diffusivity of ice - W/(m K)
dkdz  = 1.0

# initial values :
n     = 20                     # num of z-positions
zs    = 0                      # surface start
zb    = -40                    # depth - m
dt    = 20*spy                 # time-step - s
t0    = 0.0                    # begin time
tf    = 20000*spy              # end-time - s
z     = linspace(0, zb, n)     # z-coordinate corresponding to theta - m
dz    = z[1]                   # vertical step - m
theta = ones(n) * -10          # temperature - degree C
sigma = zeros(n)               # rescaled vertical component
w     = zeros(n)               # vertical ice velocity - m/a

# calculate the values for each node :
for i in range(n):
  sigma[i]  = (z[i] - z[-1]) / (z[0] - z[-1])
  w[i]      = 0.2*sigma[i] / spy     # meters per year
#==============================================================================

# Create mesh and define function space
mesh = Interval(n, zb, zs)
V = FunctionSpace(mesh, 'Lagrange', 1)

# Define boundary conditions
u0 = Constant(-10)

class Boundary(SubDomain):  # define the Dirichlet boundary
  def inside(self, x, on_boundary):
    return on_boundary

boundary = Boundary()
bc = DirichletBC(V, u0, boundary)

# Initial condition
u_1 = interpolate(u0, V)
#u_1 = project(u0, V)  # will not result in exact solution!

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
a = dt*k/(rho*cp) * inner(nabla_grad(u), nabla_grad(v))*dx #+ \
#    v*dt*nabla_grad(u)*dx
L =  (u_1 + dt/(rho*cp) * dkdz)*v*dx

A = assemble(a)   # assemble only once, before the time stepping
b = None          # necessary for memory saving assemeble call

# Compute solution
u = Function(V)   # the unknown at a new time level
t = dt
while t <= tf:
  print 'time =', t
  b = assemble(L, tensor=b)
  u0.t = t
  bc.apply(A, b)
  solve(A, u.vector(), b)

  t += dt
  u_1.assign(u)

plot(u)
interactive()



