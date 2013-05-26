from pylab import *

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

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

tits1 = ['Density', 
         'Temperature', 
         'Densification Rate', 
         'Age']
tits2 = [r'Depth at $\rho = 815$',
         r'Age at $\rho = 815$',
         r'Integrated Porosity up to $\rho = 815$',
         'Integrated Porosity of Column']
xlabs1 = [r'$\rho$', 
          r'$T$', 
          r'$\frac{d\rho}{dt}$', 
          r'$a$']
ylabs2 = [r'$z$',
          r'$a$',
          r'$\phi$',
          r'$\phi$']

def plot_all():
  for ax, u, tit, xlab in zip(axs1, us1, tits1, xlabs1):
    ax.plot(u[:,0],  z, 'k',   lw=2, label='initial')
    ax.plot(u[:,-1], z, 'r--', lw=2, label='2000 years')
    ax.set_xlabel(xlab)
    ax.set_ylabel(r'$z$')
    ax.set_ylim([-200, 0])
    ax.grid()
  for ax, u, tit, ylab in zip(axs2, us2, tits2, ylabs2):
    ax.plot(t815, u, 'k', lw=2)
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(ylab)
    ax.grid()
  ax1.legend(loc="lower left")
  ax2.legend(loc="lower right")
  ax3.legend(loc="lower right")
  ax4.legend(loc="upper right")
  savefig('plot.png', dpi=300)


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
  plot(rho[:,0],  z, 'k',   lw=2, label='initial')
  plot(rho[:,-1], z, 'k--', lw=2, label='2000 years')
  title('Density')
  xlabel(r'$\rho$')
  ylabel(r'$z$')
  legend()
  grid()
  show()

def plot_temp():
  plot(T[:,0],  z, 'k',   lw=2, label='initial')
  plot(T[:,-1], z, 'k--', lw=2, label='2000 years')
  title('Temperature')
  xlabel(r'$T$')
  ylabel(r'$z$')
  legend()
  grid()
  show()

def plot_drhodt():
  plot(drhodt[:,0],  z, 'k',   lw=2, label='initial')
  plot(drhodt[:,-1], z, 'k--', lw=2, label='2000 years')
  title('Densification Rate')
  xlabel(r'$\frac{d\rho}{dt}$')
  ylabel(r'$z$')
  legend(loc="lower right")
  grid()
  show()

def plot_age():
  plot(age[:,0],  z, 'k',   lw=2, label='initial')
  plot(age[:,-1], z, 'k--', lw=2, label='2000 years')
  title('Age')
  xlabel(r'$a$')
  ylabel(r'$z$')
  legend()
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
