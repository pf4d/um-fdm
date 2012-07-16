import os.path
from numpy import array, double
from pylab import *
from scipy.interpolate import interp1d

def give_density():
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

#s, d = give_density()
#
#x = d[0][:,0] + d[0][:,0]/(d[0][:,1] - d[0][:,0])
#x = x/100.0
#y = d[0][:,3]
#
#plot(x, y)
#
#f     = interp1d(x, y, bounds_error=False, fill_value=max(y))
#xnew  = arange(min(x), max(x)+1, 0.01)
#ynew  = f(xnew)
#
#plot(xnew, ynew, 'rx')
#
#show()



