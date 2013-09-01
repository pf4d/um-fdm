import inspect, os
from pylab import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

spy = 31556926.0

rhos    = []
Ts      = []
zs      = []
ages    = []
drhodts = []
pors    = []
rho815s = []
ts      = []
t815s   = []
por815s = []
porAlls = []
age815s = []
z815s   = []

for i in range(6):
  i = str(i+1)
  directory = "data/fmic/results/CummingsExperiment" + i + "/"
  
  rho    = genfromtxt(directory + "CummingsExperiment" + i + "Density.txt")
  T      = genfromtxt(directory + "CummingsExperiment" + i + "Temperature.txt")
  z      = genfromtxt(directory + "CummingsExperiment" + i + "Depth.txt")
  age    = genfromtxt(directory + "CummingsExperiment" + i + "Age.txt")
  drhodt = genfromtxt(directory + "CummingsExperiment" + i + "DensityRate.txt")
  por    = genfromtxt(directory + "CummingsExperiment" + i + "Porosity.txt")
  rho815 = genfromtxt(directory + "CummingsExperiment" + i + "Rho815.txt")
  
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

  rhos.append(rho)
  Ts.append(T)
  zs.append(z)
  ages.append(age)
  drhodts.append(drhodt)
  pors.append(por)
  rho815s.append(rho815)
  ts.append(t)
  t815s.append(t815)
  por815s.append(por815)
  porAlls.append(porAll)
  age815s.append(age815)
  z815s.append(z815)


def plot_all(ex, skip):
  i      = ex - 1
  rho    = rhos[i]
  T      = Ts[i]
  z      = zs[i]
  age    = ages[i]
  drhodt = drhodts[i]
  por    = pors[i]
  rho815 = rho815s[i]
  t      = ts[i]
  t815   = t815s[i]
  por815 = por815s[i]
  porAll = porAlls[i]
  age815 = age815s[i]
  z815   = z815s[i]
  n, m   = shape(rho)
  
  fig = figure(figsize=(20,15))
  ax1 = fig.add_subplot(331)
  ax2 = fig.add_subplot(332)
  ax3 = fig.add_subplot(333)
  ax4 = fig.add_subplot(334)
  ax5 = fig.add_subplot(335)
  ax6 = fig.add_subplot(336)
  ax7 = fig.add_subplot(337)
  ax8 = fig.add_subplot(338)
  
  axs1  = [ax1,ax2,ax3,ax4]
  axs2  = [ax5,ax6,ax7,ax8]
  us1   = [rho, T, drhodt, age]
  us2   = [z815, age815, por815, porAll]
  
  tits1  = ['Density', 
            'Temperature', 
            'Densification Rate', 
            'Age']
  tits2  = [r'Depth at $\rho = 815$',
            r'Age at $\rho = 815$',
            r'Integrated Porosity up to $\rho = 815$',
            'Integrated Porosity of Column']
  xlabs1 = [r'$\rho$', 
            r'$T$', 
            r'$\frac{d\rho}{dt}$', 
            r'$a$']
  ylabs2 = [r'$z_{815}$',
            r'$a_{815}$',
            r'$\phi_{815}$',
            r'$\phi$']
  filename = inspect.getframeinfo(inspect.currentframe()).filename
  home     = os.path.dirname(os.path.abspath(filename))
  direc    = home + "/images/fmic_results/plot" + str(ex) + ".png"
  pts      = where(z > -200)
  for ax, u, tit, xlab in zip(axs1, us1, tits1, xlabs1):
    ax.plot(u[:,0][pts],  z[pts], 'k',   lw=2, label='initial')
    ax.plot(u[:,-1][pts], z[pts], 'r--', lw=2, label='2000 years')
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$z$')
    ax.set_ylim([-200, 0])
    ax.grid()
  for ax, u, tit, ylab in zip(axs2, us2, tits2, ylabs2):
    ax.plot(t815[0::skip], u[0::skip], 'k', lw=2)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(ylab)
    ax.grid()
  ax1.legend(loc="lower left")
  ax2.legend(loc="lower right")
  ax3.legend(loc="lower right")
  ax4.legend(loc="upper right")
  tit = "Experiment " + str(ex)
  figtext(0.5, 0.94, tit, fontsize=50, horizontalalignment='center')
  savefig(direc, dpi=300)


def plot100_rho(ex):
  j = ex - 1
  for i in range(10):
    plot(rhos[j][:,i], z)
  title('First 100 Years of Density')
  xlabel(r'$\rho$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_temp(ex):
  j = ex - 1
  for i in range(10):
    plot(Ts[j][:,i], z)
  title('First 100 Years of Temperature')
  xlabel(r'$T$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_drhodt(ex):
  j = ex - 1
  for i in range(10):
    plot(drhodts[j][:,i], z)
  title('First 100 Years of Densification Rate')
  xlabel(r'$\frac{d\rho}{dt}$')
  ylabel(r'$z$')
  grid()
  show()

def plot100_age(ex):
  j = ex - 1
  for i in range(10):
    plot(ages[j][:,i], z)
  title('First 100 Years of Age')
  xlabel(r'$a$')
  ylabel(r'$z$')
  grid()
  show()

def plot_rho(ex):
  j = ex - 1
  plot(rhos[j][:,0],  z, 'k',   lw=2, label='initial')
  plot(rhos[j][:,-1], z, 'k--', lw=2, label='2000 years')
  title('Density')
  xlabel(r'$\rho$')
  ylabel(r'$z$')
  legend()
  grid()
  show()

def plot_temp(ex):
  j = ex - 1
  plot(Ts[j][:,0],  z, 'k',   lw=2, label='initial')
  plot(Ts[j][:,-1], z, 'k--', lw=2, label='2000 years')
  title('Temperature')
  xlabel(r'$T$')
  ylabel(r'$z$')
  legend()
  grid()
  show()

def plot_drhodt(ex):
  j = ex - 1
  plot(drhodts[j][:,0],  z, 'k',   lw=2, label='initial')
  plot(drhodts[j][:,-1], z, 'k--', lw=2, label='2000 years')
  title('Densification Rate')
  xlabel(r'$\frac{d\rho}{dt}$')
  ylabel(r'$z$')
  legend(loc="lower right")
  grid()
  show()

def plot_age(ex):
  j = ex - 1
  plot(ages[j][:,0],  z, 'k',   lw=2, label='initial')
  plot(ages[j][:,-1], z, 'k--', lw=2, label='2000 years')
  title('Age')
  xlabel(r'$a$')
  ylabel(r'$z$')
  legend()
  grid()
  show()

def plot_z815(ex, skip):
  j = ex - 1
  plot(t815s[j][0::skip], z815s[j][0::skip], label='Exp %s' % ex)
  title(r'Depth at $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$z$')
  grid()
  show() 

def plot_age815(ex, skip):
  j = ex - 1
  plot(t815s[j][0::skip], age815s[j][0::skip], label='Exp %s' % ex)
  title(r'Age at $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$a$')
  grid()
  show() 

def plot_por815(ex, skip):
  j = ex - 1
  plot(t815s[j][0::skip], por815s[j][0::skip], label='Exp %s' % ex)
  title(r'Integrated Porosity up to $\rho = 815$')
  xlabel(r'$t$')
  ylabel(r'$\phi$')
  grid()
  show() 

def plot_porAll(ex, skip):
  j = ex - 1
  plot(t815s[j][0::skip], porAlls[j][0::skip], label='Exp %s' % ex)
  title('Integrated Porosity of Column')
  xlabel(r'$t$')
  ylabel(r'$\phi$')
  grid()
  show() 
