from numpy import *
from dolfin import *


def refine_mesh(mesh, z, l, divs, dz, i, k,  m=1):
  """
  splits the mesh a given number of times.

  INPUTS:
    mesh - mesh to refine
    z    - z coordinates of mesh
    l    - cell height vector
    divs - number of times to split mesh
    dz   - cell size of surface node
    i    - fraction of the mesh from the surface to split
    k    - multiple to increase i by each step to reduce the distance from the
           surface to split
    m    - counter used to keep track of calls
  OUTPUTS:
   tuple (z, l, mesh) - refined z-coordinates, cell height vector, and mesh 
                        respectively

  """

  if m > divs :
    return z, l, mesh

  else :
    zs = z[-1]
    zb = z[0]

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(zs)
    for cell in cells(mesh):
      p  = cell.midpoint()
      if p.distance(origin) < (zs - zb) / i:
        cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    # update coordinates :
    z      = mesh.coordinates()[:,0]              # initial z-coord
    numNew = len(z) - len(l)                      # number of split nodes
    l      = l[:-numNew]                          # remove split heights
    l      = append(l, dz/2 * ones(numNew * 2))   # append new split heights

    return refine_mesh(mesh, z, l, divs, l[-1], i*k, k, m=m+1)


def set_ini_conv(H_i, rho_i, w_i, h, h_1, a, a_1):
  """
  sets the firn model's initial state based on files in data/enthalpy folder.
  """
  rhoin   = genfromtxt("data/enthalpy/rho.txt")
  win     = genfromtxt("data/enthalpy/w.txt")
  zTemp   = genfromtxt("data/enthalpy/z.txt")
  ain     = genfromtxt("data/enthalpy/a.txt")
  Hin     = genfromtxt("data/enthalpy/H.txt")
  zs_0    = zTemp[-1]

  rho_i.vector().set_local(rhoin)
  H_i.vector().set_local(Hin)
  w_i.vector().set_local(win)
  h_0 = project(as_vector([H_i,rho_i,w_i]), MV) # project inital values on space
  h.vector().set_local(h_0.vector().array())    # initalize T, rho in solution
  h_1.vector().set_local(h_0.vector().array())  # initalize T, rho in prev. sol
  a.vector().set_local(ain)
  a_1.vector().set_local(ain)
  return zs_0


def project_vars(V, H, T, rho, drhodt, a, w, k, c, omega):
  """
  Project the variables onto the space V and update firn object.
  """
  Hplot      = project(H, V).vector().array()
  Tplot      = project(T, V).vector().array()
  rhoplot    = project(rho, V).vector().array()
  drhodtplot = project(drhodt, V).vector().array()
  aplot      = a.vector().array()
  wplot      = project(w, V).vector().array()
  kplot      = project(k, V).vector().array()
  cplot      = project(c, V).vector().array()

  return (Hplot, Tplot, rhoplot, drhodtplot, aplot, wplot, kplot, cplot, omega)
