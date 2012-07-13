"""
plot.py
Evan Cummings
07.09.12

Plotting for Firn Densification Model.

"""

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import numpy as np

class FixedOrderFormatter(ScalarFormatter):
  """Formats axis ticks using scientific notation with a constant order of 
  magnitude"""
  def __init__(self, order_of_mag=0, useOffset=True, useMathText=False):
    self._order_of_mag = order_of_mag
    ScalarFormatter.__init__(self, useOffset=useOffset, 
                             useMathText=useMathText)
  def _set_orderOfMagnitude(self, range):
    """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
    self.orderOfMagnitude = self._order_of_mag


class firn():
  """
  Data structure to hold firn model data.
  """
  def __init__(self, T, rho, w, k1, k2, k3, z, index, zb, zs):

    self.T     = T 
    self.rho   = rho
    self.w     = w
    self.k1    = k1
    self.k2    = k2
    self.k3    = k3
    self.z     = z
    self.index = index
    self.zb    = zb
    self.zs    = zs
    self.origZ = zs


class plot():
  """
  Plotting class handles all things related to plotting.
  """
  def __init__(self, firn):
    """
    Initialize plots with firn object as input.
    """    
    # x-values :
    self.firn = firn
    T         = firn.T
    rho       = firn.rho
    w         = firn.w
    k1        = firn.k1
    k2        = firn.k2
    k3        = firn.k3

    # y-value :
    z         = firn.z[firn.index]
    zs        = firn.zs
    zb        = firn.zb
    index     = firn.index
    
    # original surface height :
    origZ     = firn.origZ

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

    zmax   = zs + 20                            # max z-coord
    zmin   = zb                                 # min z-coord

    self.fig   = plt.figure(figsize=(16,6))
    self.Tax   = self.fig.add_subplot(141)
    self.rhoax = self.fig.add_subplot(142)
    self.wax   = self.fig.add_subplot(143)
    self.kax   = self.fig.add_subplot(144)

    # format : [xmin, xmax, ymin, ymax]
    self.Tax.axis([Tmin, Tmax, zmin, zmax])
    self.Tax.grid()
    self.rhoax.axis([rhoMin, rhoMax, zmin, zmax])
    self.rhoax.grid()
    self.rhoax.xaxis.set_major_formatter(FixedOrderFormatter(2))
    self.wax.axis([wMin, wMax, zmin, zmax])
    self.wax.grid()
    self.wax.xaxis.set_major_formatter(FixedOrderFormatter(-6))
    self.kax.axis([kMin, kMax, zmin, zmax])
    self.kax.grid()

    # plots :
    self.phT,     = self.Tax.plot(T - 273.15, z, 'r-')
    self.phTs,    = self.Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)
    self.phTs_0,  = self.Tax.plot(Th, origZ, 'ko')
    self.phTsp,   = self.Tax.plot(Th*np.ones(len(z)), z, 'r+')

    self.phrho,   = self.rhoax.plot(rho, z, 'g-')
    self.phrhoS,  = self.rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2)
    self.phrhoS_0,= self.rhoax.plot(rhoh, origZ, 'ko')
    self.phrhoSp, = self.rhoax.plot(rhoh*np.ones(len(z)), z, 'r+')

    self.phw,     = self.wax.plot(w, z, 'b-')
    self.phws,    = self.wax.plot([wMin, wMax], [zs, zs], 'k-', lw=2)
    self.phws_0,  = self.wax.plot(wh, origZ, 'ko')
    self.phwsp,   = self.wax.plot(wh*np.ones(len(z)), z, 'r+')

    self.phk1,    = self.kax.plot(k1, z, label='Van Dusen')
    self.phk2,    = self.kax.plot(k2, z, label='Arthern')
    self.phk3,    = self.kax.plot(k3, z, label='Schwerdtfeger' )
    self.phks,    = self.kax.plot([kMin, kMax], [zs, zs], 'k-', lw=2)
    self.phks_0,  = self.kax.plot(kh, origZ, 'ko')
    self.phksp,   = self.kax.plot(kh*np.ones(len(z)), z, 'r+')

    # formatting :
    self.fig_text = plt.figtext(.85,.95,'Time = 0.0 yr')

    self.Tax.set_title('Temperature')
    self.Tax.set_xlabel(r'$T$ $[\degree C]$')
    self.Tax.set_ylabel(r'Depth $[m]$')

    self.rhoax.set_title('Density')
    self.rhoax.set_xlabel(r'$\rho$ $\left [\frac{kg}{m^3}\right ]$')
    #self.rhoax.set_ylabel(r'Depth $[m]$')

    self.wax.set_title('Velocity')
    self.wax.set_xlabel(r'$w$ $\left [\frac{mm}{s}\right ]$')
    #self.wax.set_ylabel(r'Depth $[m]$')

    self.kax.set_title('Thermal Conductivity')
    self.kax.set_xlabel(r'$k$ $\left [\frac{J}{m K s} \right ]$')
    #self.kax.set_ylabel(r'Depth $[m]$')
    
    # Legend formatting:
    leg    = self.kax.legend(loc='lower center')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)


  def update_plot(self, t):
    """
    Update the plot for each time step at time t.
    """    
    T     = self.firn.T
    rho   = self.firn.rho
    w     = self.firn.w
    k1    = self.firn.k1
    k2    = self.firn.k2
    k3    = self.firn.k3
    z     = self.firn.z
    origZ = self.firn.origZ
    index = self.firn.index

    self.fig_text.set_text('Time = %.2f yr' % t) 
    self.phT.set_xdata(T[index] - 273.15)
    self.phT.set_ydata(z)
    self.phTs.set_ydata(z[-1])
    self.phTs_0.set_ydata(origZ)
    self.phTsp.set_ydata(z)
    
    self.phrho.set_xdata(rho[index])
    self.phrho.set_ydata(z)
    self.phrhoS.set_ydata(z[-1])
    self.phrhoS_0.set_ydata(origZ)
    self.phrhoSp.set_ydata(z)
   
    self.phw.set_xdata(w[index])
    self.phw.set_ydata(z)
    self.phws.set_ydata(z[-1])
    self.phws_0.set_ydata(origZ)
    self.phwsp.set_ydata(z)
    
    self.phk1.set_xdata(k1[index])
    self.phk2.set_xdata(k2[index])
    self.phk3.set_xdata(k3[index])
    self.phk1.set_ydata(z)
    self.phk2.set_ydata(z)
    self.phk3.set_ydata(z)
    self.phks.set_ydata(z[-1])
    self.phks_0.set_ydata(origZ)
    self.phksp.set_ydata(z)


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
      Tax.plot(firn.T[firn.index] - 273.15, firn.z, color, label=title, lw=2)
      Tax.plot([Tmin, Tmax], [firn.z[-1], firn.z[-1]], color, lw=2)

      rhoax.plot(firn.rho[firn.index], firn.z, color, lw=2)
      rhoax.plot([rhoMin, rhoMax], [firn.z[-1], firn.z[-1]], color, lw=2)

      wax.plot(firn.w[firn.index], firn.z, color, lw=2)
      wax.plot([wMin, wMax], [firn.z[-1], firn.z[-1]], color, lw=2)

      kax.plot(firn.k2[firn.index], firn.z, color, lw=2)
      kax.plot([kMin, kMax], [firn.z[-1], firn.z[-1]], color, lw=2)
    
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
    plt.plot(x, ht,     label='Surface Height')
    plt.plot(x, origHt, label='Original Surface')
    plt.xlabel(r'time $[a]$')
    plt.ylabel(r'height $[m]$')
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
    
    ax.set_xlabel(r'time $[a]$')
    ax.set_ylabel(r'height $[m]$')
    ax.set_title('Surface Height Changes')
    ax.grid()
  
    # Legend formatting:
    leg    = ax.legend(loc='lower left')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)
    plt.show()



