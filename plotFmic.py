"""
enthPlot.py
Evan Cummings
07.09.12

Plotting for enthalby Firn Densification Model.

"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from pylab import mpl
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


class firn():
  """
  Data structure to hold firn model state data.
  """
  def __init__(self, H, T, rho, drhodt, rho815, a, 
               por, omega, w, k, c, z, index):

    self.H      = H
    self.T      = T 
    self.rho    = rho
    self.drhodt = drhodt
    self.rho815 = rho815
    self.a      = a
    self.por    = por
    self.omega  = omega
    self.w      = w
    self.k      = k
    self.c      = c
    self.z      = z[index]
    self.index  = index
    self.zb     = z[index][0]
    self.zs     = z[index][-1]
    self.origZ  = self.zs
    self.Ts     = H[-1] / c[-1]


class plot():
  """
  Plotting class handles all things related to plotting.
  """
  def __init__(self, firn):
    """
    Initialize plots with firn object as input.
    """   
    self.spy  = 31556926.0
     
    # x-values :
    T      = firn.T
    rho    = firn.rho
    w      = firn.w * self.spy * 1e2 # cm/a
    a      = firn.a/self.spy
    Ts     = firn.Ts - 273.15

    # y-value :
    z      = firn.z
    zs     = firn.zs
    zb     = firn.zb
    
    # original surface height :
    origZ  = firn.origZ

    zmax   = zs + (zs - zb) / 5                   # max z-coord
    zmin   = zb                                   # min z-coord

    Tmin   = -65                                  # T x-coord min
    Tmax   = -35                                  # T x-coord max
    Th     = Tmin + 0.1*(Tmax - Tmin) / 2         # T height x-coord
    Tz     = zmax - 0.15*(zmax - zmin) / 2        # z-coord of Ts

    rhoMin = 300                                  # rho x-coord min
    rhoMax = 1000                                 # rho x-coord max
    #rhoh   = rhoMin + 0.1*(rhoMax - rhoMin) / 2  # rho height x-coord
    
    wMin   = -28
    wMax   = 0
    wh     = wMin + 0.1*(wMax - wMin) / 2

    aMin   = 0.0
    aMax   = 10000.0
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
    self.phT,     = self.Tax.plot(T - 273.15, z, '0.3', lw=1.2)
    self.phTs,    = self.Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)
    self.phTs_0,  = self.Tax.plot(Th, origZ, 'ko')
    self.phTsp,   = self.Tax.plot(Th*np.ones(len(z)), z, 'r+')
    
    self.phrho,   = self.rhoax.plot(rho, z, '0.3', lw=1.2)
    self.phrhoS,  = self.rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2)
    #self.phrhoS_0,= self.rhoax.plot(rhoh, origZ, 'ko')
    #self.phrhoSp, = self.rhoax.plot(rhoh*np.ones(len(z)), z, 'r+')

    self.phw,     = self.wax.plot(w, z, '0.3', lw=1.2)
    self.phwS,    = self.wax.plot([wMin, wMax], [zs, zs], 'k-', lw=2)
    #self.phws_0,  = self.wax.plot(wh, origZ, 'ko')
    #self.phwsp,   = self.wax.plot(wh*np.ones(len(z)), z, 'r+')
    
    self.pha,     = self.aax.plot(a, z, '0.3', lw=1.2)
    self.phaS,    = self.aax.plot([aMin, aMax], [zs, zs], 'k-', lw=2)
    #self.phks_0,  = self.kax.plot(kh, origZ, 'ko')
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
    origZ = firn.origZ
    Ts    = firn.Ts - 273.15

    self.fig_text.set_text('Time = %.2f yr' % t) 
    
    self.Tsurf.set_text(r'Surface Temp: %.1f $\degree$C' % Ts)
    self.phT.set_xdata(T - 273.15)
    self.phT.set_ydata(z)
    self.phTs.set_ydata(z[-1])
    self.phTs_0.set_ydata(origZ)
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



