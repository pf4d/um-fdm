"""
enthPlot.py
Evan Cummings
07.09.12

Plotting for enthalby Firn Densification Model.

"""
from numpy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from pylab import mpl
from scipy.interpolate import interp1d
from dolfin import *
import numpy as np

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'
    
class FixedOrderFormatter(ScalarFormatter):
  """
  Formats axis ticks using scientific notation with a constant order of 
  magnitude
  """
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                             useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """
    Over-riding this to avoid having orderOfMagnitude reset elsewhere
    """
    self.orderOfMagnitude = self._order_of_mag


class Constants():
  
  def __init__(self):
    """
    Data structure for constants.
    """
    self.pi    = 3.141592653589793 # pi ............................. rad
    self.g     = 9.81              # gravitational acceleration ..... m/s^2
    self.R     = 8.3144621         # gas constant ................... J/(mol K)
    self.spy   = 31556926.0        # seconds per year ............... s/a
    self.rhoi  = 917.              # density of ice ................. kg/m^3
    self.rhoin = self.rhoi         # initial density of column ...... kg/m^3
    self.rhow  = 1000.             # density of water ............... kg/m^3
    self.rhom  = 550.              # density at 15 m ................ kg/m^3
    self.rhoc  = 815.              # density at critical value ...... kg/m^3
    self.ki    = 2.1               # thermal conductivity of ice .... W/(m K)
    self.cpi   = 2009.             # const. heat capacitity of ice .. J/(kg K)
    self.kcHh  = 3.7e-9            # creep coefficient high ......... (m^3 s)/kg
    self.kcLw  = 9.2e-9            # creep coefficient low .......... (m^3 s)/kg
    self.kg    = 1.3e-7            # grain growth coefficient ....... m^2/s  
    self.Ec    = 60e3              # act. energy for water in ice ... J/mol
    self.Eg    = 42.4e3            # act. energy for grain growth ... J/mol
    self.Tw    = 273.15            # triple point water ............. degrees K
    self.T0    = 0.0               # reference temperature .......... K
    self.beta  = 7.9e-8            # Clausius-Clapeyron ............. K/Pa
    self.Lf    = 3.34e5            # latent heat of fusion .......... J/kg
    self.Hsp   = self.cpi*self.Tw  # Enthalpy of ice at Tw .......... J/kg


class Firn():
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, const, FEMdata, data, Tavg, 
               rhos, adot, A, acc, z, l, index, dt):

    self.const   = const                     # constants

    self.mesh    = FEMdata[0]                # mesh
    self.V       = FEMdata[1]                # function space
    self.MV      = FEMdata[2]                # Mixed function space
    self.H_i     = FEMdata[3]                # initial enthalpy
    self.rho_i   = FEMdata[4]                # initial density
    self.w_i     = FEMdata[5]                # initial velocity
    self.a_i     = FEMdata[6]                # initial age
    self.h       = FEMdata[7]                # enthalpy, density, velocity
    self.HF      = FEMdata[8]                # enthalpy
    self.TF      = FEMdata[9]                # temperature
    self.rhoF    = FEMdata[10]               # density
    self.drhodtF = FEMdata[11]               # densification rate
    self.wF      = FEMdata[12]               # velocity
    self.aF      = FEMdata[13]               # age
    self.h_1     = FEMdata[14]               # previous step's solution
    self.a_1     = FEMdata[15]               # previous step's age
    self.kF      = FEMdata[16]               # thermal conductivity
    self.cF      = FEMdata[17]               # heat capacity

    self.H       = data[0]                   # enthalpy
    self.T       = data[1]                   # temperature
    self.rho     = data[2]                   # density
    self.drhodt  = data[3]                   # densification rate
    self.a       = data[4]                   # age
    self.w       = data[5]                   # vertical velocity
    self.k       = data[6]                   # thermal conductivity
    self.c       = data[7]                   # heat capacity
    self.omega   = data[8]                   # percentage of water content

    self.Tavg    = Tavg                      # average surface temperature
    self.rhos    = rhos                      # initial density at surface
    self.adot    = adot                      # accumulation rate
    self.A       = A                         # surface accumulation
    self.acc     = acc                       # surface accumulation
    self.z       = z[index]                  # z-coordinates of mesh
    self.l       = l                         # height vector
    self.lini    = l                         # initial height vector
    self.index   = index                     # index of ordered, refined mesh
    self.dt      = dt                        # time step
    
    self.n       = len(self.H)               # system DOF
    self.domega  = np.zeros(self.n)          # percent change of water content
    self.rhoin   = self.rho                  # initial density vector
    self.zb      = z[index][0]               # base of firn
    self.zs      = z[index][-1]              # surface of firn
    self.zs_1    = self.zs                   # previous time-step surface  
    self.zo      = self.zs                   # z-coordinate of initial surface
    self.ht      = [self.zs]                 # list of surface heights
    self.origHt  = [self.zo]                 # list of initial surface heights
    self.Ts      = self.H[-1] / self.c[-1]   # temperature of surface

    self.porAll  = 0.0                       # porosity of column
    self.por815  = 0.0                       # porosity to rhoc
    self.z815    = 0.0                       # depth of rhoc
    self.age815  = 0.0                       # age of rhoc


  def update_vars(self):
    """
    Project the variables onto the space V and update firn object.
    """
    self.H      = project(self.HF, self.V).vector().array()
    self.T      = project(self.TF, self.V).vector().array()
    self.rho    = project(self.rhoF, self.V).vector().array()
    self.drhodt = project(self.drhodtF, self.V).vector().array()
    self.a      = self.aF.vector().array()
    self.w      = project(self.wF, self.V).vector().array()
    self.k      = project(self.kF, self.V).vector().array()
    self.c      = project(self.cF, self.V).vector().array()
    
    self.Ts     = self.H[-1] / self.c[-1]
    self.acc    = self.const.rhoi*self.adot/self.const.spy
    self.A      = self.const.spy*self.acc/self.rhos*1e3


  def save_state(self):
    """
    Save the current state of firn object to /data/enthalpy directory.
    """
    savetxt("data/umcur/init/z.txt",   self.z)
    savetxt("data/umcur/init/l.txt",   self.l)
    savetxt("data/umcur/init/a.txt",   self.a)
    savetxt("data/umcur/init/rho.txt", self.rho)
    savetxt("data/umcur/init/H.txt",   self.H)
    savetxt("data/umcur/init/w.txt",   self.w)
    print "saved the current state of firn"
  
  
  def update_height_history(self):
    """
    track the current height of the firn :
    """
    self.ht.append(self.z[-1])

    # calculate the new height of original surface by interpolating the 
    # vertical speed from w and keeping the ratio intact :
    interp     = interp1d(self.z, self.w,
                          bounds_error=False,
                          fill_value=self.w[0])
    zint       = np.array([self.zo])
    wzo        = interp(zint)[0]
    dt         = self.dt
    zs         = self.z[-1]
    zb         = self.z[0]
    zs_1       = self.zs_1
    zo         = self.zo
    self.zo    = (zs - zb) * (zo - zb) / (zs_1 - zb) + wzo * dt
    
    # track original height :
    if self.zo > zb:
      self.origHt.append(self.zo)
    
    # update the previous time steps' surface height :
    self.zs_1  = self.z[-1]


  def update_height(self):
    """
    If conserving the mass of the firn column, calculate height of each 
    interval :
    """
    lnew     = self.lini*self.rhoin / self.rho
    zSum     = self.zb
    zTemp    = np.zeros(self.n)
    for i in range(self.n)[1:]:
      zTemp[i] = zSum + lnew[i]
      zSum    += lnew[i]
    self.z   = zTemp
    self.l   = lnew
    self.mesh.coordinates()[:,0][self.index] = self.z  # update the mesh coord.


  def adjust_vectors(self, Kcoef, Tcoef, rhoCoef):
    """
    Adjust the vectors for enthalpy and density.
    """
    n    = self.n
    kcHh = self.const.kcHh 
    kcLw = self.const.kcLw 
    Hsp  = self.const.Hsp  
    Tw   = self.const.Tw   
    T0   = self.const.T0   
    Lf   = self.const.Lf   
    rhow = self.const.rhow

    # find vector of T, rho :
    self.H      = project(self.HF, self.V).vector().array()
    self.rho    = project(self.rhoF, self.V).vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    rhoCoefNew          = ones(n)
    rhoHigh             = where(self.rho >  550)[0]
    rhoLow              = where(self.rho <= 550)[0]
    rhoCoefNew[rhoHigh] = kcHh*(2.366 - 0.293*np.log(self.A))
    rhoCoefNew[rhoLow]  = kcLw*(1.435 - 0.151*np.log(self.A))
    rhoCoef.vector().set_local(rhoCoefNew)
  
    # update coefficients used by enthalpy :
    Hhigh               = where(self.H >= Hsp)[0]
    Hlow                = where(self.H <  Hsp)[0]
    omegaNew            = zeros(n)
    TcoefNew            = ones(n)
    KcoefNew            = ones(n)
  
    KcoefNew[Hhigh]     = 1/10.0
    TcoefNew[Hhigh]     = self.c[Hhigh] / self.H[Hhigh] * Tw
  
    # update water content and density :
    omegaNew[Hhigh]     = (self.H[Hhigh] - self.c[Hhigh]*(Tw - T0)) / Lf
    domega              = omegaNew - self.omega          # water content chg.
    self.domega         = domega
    domPos              = where(domega >  0)[0]          # water content inc.
    domNeg              = where(domega <= 0)[0]          # water content dec.
    rhoNotLiq           = where(self.rho < rhow)[0]      # density < water
    rhoInc              = intersect1d(domPos, rhoNotLiq) # where rho can inc.
    self.rho[rhoInc]    = self.rho[rhoInc] + domega[rhoInc]*rhow 
    self.rho[domNeg]    = self.rho[domNeg] + domega[domNeg]*83.0
  
    # update the dolfin vectors :
    self.rho_i.vector().set_local(self.rho)
    h_0 = project(as_vector([self.HF, self.rho_i, self.wF]), self.MV)
    self.h.vector().set_local(h_0.vector().array())
    #Kcoef.vector().set_local(KcoefNew)  #FIXME: erratic 
    Tcoef.vector().set_local(TcoefNew)


  def set_ini_conv(self):
    """
    sets the firn model's initial state based on files in data/enthalpy folder.
    """
    self.rhoin = genfromtxt("data/umcur/init/rho.txt")
    self.rho   = self.rhoin                  
    self.w     = genfromtxt("data/umcur/init/w.txt")
    self.z     = genfromtxt("data/umcur/init/z.txt")
    self.a     = genfromtxt("data/umcur/init/a.txt")
    self.H     = genfromtxt("data/umcur/init/H.txt")
    self.lin   = genfromtxt("data/umcur/init/l.txt")
    
    self.zs_1    = self.z[-1]                # previous time-step surface  
    self.zo      = self.z[-1]                # z-coordinate of initial surface
    self.ht      = [self.z[-1]]              # list of surface heights
    self.origHt  = [self.z[-1]]              # list of initial surface heights
    self.Ts      = self.H[-1] / self.c[-1]   # temperature of surface
  
    self.rho_i.vector().set_local(self.rho)
    self.H_i.vector().set_local(self.H)
    self.w_i.vector().set_local(self.w)
    h_0 = project(as_vector([self.H_i,self.rho_i,self.w_i]), self.MV)
    self.h.vector().set_local(h_0.vector().array())
    self.h_1.vector().set_local(h_0.vector().array())
    self.aF.vector().set_local(self.a)
    self.a_1.vector().set_local(self.a)


class Plot():
  """
  Plotting class handles all things related to plotting.
  """
  def __init__(self, firn):
    """
    Initialize plots with firn object as input.
    """   
    self.spy  = firn.const.spy
    Tw        = firn.const.Tw
     
    # x-values :
    T      = firn.T
    rho    = firn.rho
    w      = firn.w * self.spy * 1e2 # cm/a
    a      = firn.a/self.spy
    Ts     = firn.Ts - Tw

    # y-value :
    z      = firn.z
    zs     = firn.z[-1]
    zb     = firn.z[0]
    
    # original surface height :
    zo     = firn.zo

    zmax   = zs + (zs - zb) / 5                   # max z-coord
    zmin   = zb                                   # min z-coord

    Tmin   = firn.Tavg - Tw - 15                  # T x-coord min
    Tmax   = firn.Tavg - Tw + 15                  # T x-coord max
    Th     = Tmin + 0.1*(Tmax - Tmin) / 2         # T height x-coord
    Tz     = zmax - 0.15*(zmax - zmin) / 2        # z-coord of Ts

    rhoMin = 300                                  # rho x-coord min
    rhoMax = 1000                                 # rho x-coord max
    #rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord
    
    wMin   = -30
    wMax   = 0
    wh     = wMin + 0.1*(wMax - wMin) / 2

    aMin   = 0.0
    aMax   = 1400.0
    #kh     = kMin + 0.1*(kMax - kMin) / 2

    self.fig   = plt.figure(figsize=(15,6))
    self.Tax   = self.fig.add_subplot(141)
    self.rhoax = self.fig.add_subplot(142)
    self.wax   = self.fig.add_subplot(143)
    self.aax   = self.fig.add_subplot(144)

    # format : [xmin, xmax, ymin, ymax]
    self.Tax.axis([Tmin, Tmax, zmin, zmax])
    self.Tax.grid()
    self.rhoax.axis([rhoMin, rhoMax, zmin, zmax])
    self.rhoax.grid()
    self.rhoax.xaxis.set_major_formatter(FixedOrderFormatter(2))
    self.wax.axis([wMin, wMax, zmin, zmax])
    self.wax.grid()
    self.aax.axis([aMin, aMax, zmin, zmax])
    self.aax.xaxis.set_major_formatter(FixedOrderFormatter(3))
    self.aax.grid()

    # plots :
    self.Tsurf    = self.Tax.text(Th, Tz, r'Surface Temp: %.1f $\degree$C' % Ts)
    self.phT,     = self.Tax.plot(T - 273.15, z, '0.3', lw=2)
    self.phTs,    = self.Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)
    self.phTs_0,  = self.Tax.plot(Th, zo, 'ko')
    self.phTsp,   = self.Tax.plot(Th*np.ones(len(z)), z, 'r+')
    
    self.phrho,   = self.rhoax.plot(rho, z, '0.3', lw=2)
    self.phrhoS,  = self.rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2)
    #self.phrhoS_0,= self.rhoax.plot(rhoh, zo, 'ko')
    #self.phrhoSp, = self.rhoax.plot(rhoh*np.ones(len(z)), z, 'r+')

    self.phw,     = self.wax.plot(w, z, '0.3', lw=2)
    self.phwS,    = self.wax.plot([wMin, wMax], [zs, zs], 'k-', lw=2)
    #self.phws_0,  = self.wax.plot(wh, zo, 'ko')
    #self.phwsp,   = self.wax.plot(wh*np.ones(len(z)), z, 'r+')
    
    self.pha,     = self.aax.plot(a, z, '0.3', lw=2)
    self.phaS,    = self.aax.plot([aMin, aMax], [zs, zs], 'k-', lw=2)
    #self.phks_0,  = self.kax.plot(kh, zo, 'ko')
    #self.phksp,   = self.kax.plot(kh*np.ones(len(z)), z, 'r+')

    # formatting :
    self.fig_text = plt.figtext(.85,.95,'Time = 0.0 yr')

    self.Tax.set_title('Temperature')
    self.Tax.set_xlabel(r'$T\ [\degree \mathrm{C}]$')
    self.Tax.set_ylabel('Depth [m]')

    self.rhoax.set_title('Density')
    self.rhoax.set_xlabel(r'$\rho\ \left[\frac{\mathrm{kg}}{\mathrm{m}^3}\right]$')
    
    self.wax.set_title('Velocity')
    self.wax.set_xlabel(r'$w\ \left[\frac{\mathrm{cm}}{\mathrm{a}}\right]$')

    self.aax.set_title('Age')
    self.aax.set_xlabel(r'$a\ [\mathrm{a}]$')
    

  def update_plot(self, firn, t):
    """
    Update the plot for each time step at time t.
    """    
    T     = firn.T
    rho   = firn.rho
    w     = firn.w * self.spy * 1e2
    a     = firn.a/self.spy
    z     = firn.z
    zo    = firn.zo
    Ts    = firn.Ts - 273.15

    self.fig_text.set_text('Time = %.2f yr' % t) 
    
    self.Tsurf.set_text(r'Surface Temp: %.1f $\degree$C' % Ts)
    self.phT.set_xdata(T - 273.15)
    self.phT.set_ydata(z)
    self.phTs.set_ydata(z[-1])
    self.phTs_0.set_ydata(zo)
    self.phTsp.set_ydata(z)
    
    self.phrho.set_xdata(rho)
    self.phrho.set_ydata(z)
    self.phrhoS.set_ydata(z[-1])
    
    self.phw.set_xdata(w)
    self.phw.set_ydata(z)
    self.phwS.set_ydata(z[-1])
   
    self.pha.set_xdata(a)
    self.pha.set_ydata(z)
    self.phaS.set_ydata(z[-1])
    

  def plot_all(self, firns, titles, colors):
    """
    Plot the data from a list of firn objects with corresponding titles and
    colors array.
    """    
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
    
    zmax   = firns[0].zs + 20                    # max z-coord
    zmin   = firns[0].zb                         # min z-coord

    fig    = plt.figure(figsize=(16,6))
    Tax    = fig.add_subplot(141)
    rhoax  = fig.add_subplot(142)
    wax    = fig.add_subplot(143)
    kax    = fig.add_subplot(144)

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

    # plots :
    for firn, title, color in zip(firns, titles, colors):
      i = firn.index
      Tax.plot(firn.T[i] - 273.15, firn.z[i], color, label=title, lw=2)
      Tax.plot([Tmin, Tmax], [firn.z[i][-1], firn.z[i][-1]], color, lw=2)

      rhoax.plot(firn.rho[i], firn.z[i], color, lw=2)
      rhoax.plot([rhoMin, rhoMax], [firn.z[i][-1], firn.z[i][-1]], color, lw=2)

      wax.plot(firn.w[i], firn.z[i], color, lw=2)
      wax.plot([wMin, wMax], [firn.z[i][-1], firn.z[i][-1]], color, lw=2)

      kax.plot(firn.k2[i], firn.z[i], color, lw=2)
      kax.plot([kMin, kMax], [firn.z[i][-1], firn.z[i][-1]], color, lw=2)
    
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
    leg    = Tax.legend(loc='upper right')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)

    plt.show()


  def plot_height(self, x, ht, origHt):
    """
    Plot the height history of a column of firn for times x, current height ht, 
    and original surface height origHt.
    """
    # plot the surface height information :
    plt.plot(x,               ht,     'k-',  lw=1.5, label=r'Surface Height')
    plt.plot(x[:len(origHt)], origHt, 'k--', lw=1.5, label=r'Original Surface')
    plt.xlabel('time [a]')
    plt.ylabel('height [m]')
    plt.title('Surface Height Changes')
    plt.grid()
  
    # Legend formatting:
    leg = plt.legend(loc='lower left')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)
    plt.show()


  def plot_all_height(self, xs, hts, origHts, titles, colors):
    """
    Plot the height history of a list of firn objects for times array xs, 
    current height array hts, original surface heights origHts, with 
    corresponding titles and colors arrays.
    """
    zMin = min(min(origHts))
    zMax = max(max(hts))
    zMax = zMax + (zMax - zMin) / 16.0
    xMin = min(min(xs))
    xMax = max(max(xs))
    
    fig = plt.figure(figsize=(11,8))
    ax  = fig.add_subplot(111)
    
    # format : [xmin, xmax, ymin, ymax]
    ax.axis([xMin, xMax, zMin, zMax])
    ax.grid()
    
    # plot the surface height information :
    for x, ht, origHt, title, color in zip(xs, hts, origHts, titles, colors):
      ax.plot(x, ht,     color + '-',  label=title + ' Surface Height')
      ax.plot(x, origHt, color + '--', label=title + ' Original Surface')
    
    ax.set_xlabel('time [a]')
    ax.set_ylabel('height [m]')
    ax.set_title('Surface Height Changes')
    ax.grid()
  
    # Legend formatting:
    leg    = ax.legend(loc='lower left')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)
    plt.show()



