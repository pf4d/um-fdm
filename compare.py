from pylab import *

spy    = 31556926.0

ex = sys.argv[1]
ex = str(ex)

directory = "data/fmic/CummingsExperiment" + ex + "/"

rho    = genfromtxt(directory + "CummingsExperiment" + ex + "Density.txt")
T      = genfromtxt(directory + "CummingsExperiment" + ex + "Temperature.txt")
z      = genfromtxt(directory + "CummingsExperiment" + ex + "Depth.txt")
age    = genfromtxt(directory + "CummingsExperiment" + ex + "Age.txt")
drhodt = genfromtxt(directory + "CummingsExperiment" + ex + "DensityRate.txt")
por    = genfromtxt(directory + "CummingsExperiment" + ex + "Porosity.txt")
rho815 = genfromtxt(directory + "CummingsExperiment" + ex + "Rho815.txt")


t      = rho[0]
z      = z[1:,0]
rho    = rho[1:]
T      = T[1:]
age    = age[1:]
drhodt = drhodt[1:]


t815   = por[0]
por815 = por[1]
porAll = por[2]

t815   = rho815[0]
age815 = rho815[1]
z815   = rho815[2]


n, m   = shape(rho)


def plot100_rho():
  for i in range(10):
    plot(rho[:,i], z)
  title('First 100 Years of Density')
  xlabel(r'$\rho$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_temp():
  for i in range(10):
    plot(T[:,i], z)
  title('First 100 Years of Temperature')
  xlabel(r'$T$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_drhodt():
  for i in range(10):
    plot(drhodt[:,i], z)
  title('First 100 Years of Densification Rate')
  xlabel(r'$\frac{d\rho}{dt}$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_age():
  for i in range(10):
    plot(age[:,i], z)
  title('First 100 Years of Age')
  xlabel(r'$a$')
  ylabel(r'$z$')
  grid()
  show()

def plot_rho():
  for i in range(m):
    plot(rho[:,i], z)
  title('Density')
  xlabel(r'$\rho$')
  ylabel(r'$z$')
  grid()
  show()

def plot_temp():
  for i in range(m):
    plot(T[:,i], z)
  title('Temperature')
  xlabel(r'$T$')
  ylabel(r'$z$')
  grid()
  show()

def plot_drhodt():
  for i in range(m):
    plot(drhodt[:,i], z)
  title('Densification Rate')
  xlabel(r'$\frac{d\rho}{dt}$')
  ylabel(r'$z$')
  grid()
  show()

def plot_age():
  for i in range(m):
    plot(age[:,i], z)
  title('Age')
  xlabel(r'$a$')
  ylabel(r'$z$')
  grid()
  show()

def plot_z815():
  plot(t815, z815)
  title(r'Depth at $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$z$')
  grid()
  show() 

def plot_age815():
  plot(t815, age815)
  title(r'Age at $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$a$')
  grid()
  show() 

def plot_por815():
  plot(t815, por815)
  title(r'Integrated Porosity up to $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$\phi$')
  grid()
  show() 

def plot_porAll():
  plot(t815, porAll)
  title('Integrated Porosity of Column')
  xlabel(r'$t$')
  ylabel(r'$\phi$')
  grid()
  show() 
