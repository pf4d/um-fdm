import os.path
from scipy.interpolate import interp1d
from numpy import *
import numpy as np
from dolfin import *


def refine_mesh(mesh, divs, i, k,  m=1):
  """
  splits the mesh a given number of times.

  INPUTS:
    mesh - mesh to refine
    divs - number of times to split mesh
    i    - fraction of the mesh from the surface to split
    k    - multiple to increase i by each step to reduce the distance from the
           surface to split
    m    - counter used to keep track of calls
  OUTPUTS:
   tuple (z, l, mesh, index) - refined z-coordinates, cell height vector, 
                               mesh, and index of sorted mesh respectively

  """
  z     = mesh.coordinates()[:,0]
  index = argsort(z)

  if m > divs :
    l     = np.diff(z[index])
    return z, l, mesh, index

  else :
    zs = z[index][-1]
    zb = z[index][0]

    cell_markers = CellFunction("bool", mesh)
    cell_markers.set_all(False)
    origin = Point(zs)
    for cell in cells(mesh):
      p  = cell.midpoint()
      if p.distance(origin) < (zs - zb) / i:
        cell_markers[cell] = True
    mesh = refine(mesh, cell_markers)

    return refine_mesh(mesh, divs, i*k, k, m=m+1)


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


def give_density():
  """
  get the density, use like :

    s, d = give_density()
    
    x = d[0][:,0] + d[0][:,0]/(d[0][:,1] - d[0][:,0])
    x = x/100.0
    y = d[0][:,3]
    
    plot(x, y)
    
    f     = interp1d(x, y, bounds_error=False, fill_value=max(y))
    xnew  = arange(min(x), max(x)+1, 0.01)
    ynew  = f(xnew)
    
    plot(xnew, ynew, 'rx')
    
    show()  

  """
  f = open ('../../ice/data/OP60_CoreData/AllCores_Mass_Copy.txt', 'r')
  header = f.readline()

  data     = []
  stations = []
  temp     = []
  
  first = f.readline().split()
  temp.append(double(first[1:]))
  stations.append(first[0])

  for line in f.readlines():
    line    = line.split()
    station = line[0]
    dv      = double(line[1:])      # convert data values to doubles
    if station == stations[-1]:
      temp.append(dv)
    else:
      stations.append(station)
      data.append(array(temp))
      temp = []
    
  data.append(array(temp))

  return stations, array(data)





