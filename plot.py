"""
model.py
Evan Cummings
05.23.12

FEniCS solution to firn temperature profile.

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



class plot():

  def __init__(self, T, rho, w, k1, k2, k3, z, index, zb, zs):
    
    # x-values :
    self.T    = T
    self.rho  = rho
    self.w    = w
    self.k1   = k1
    self.k2   = k2
    self.k3   = k3

    # y-value :
    self.z     = z[index]
    self.index = index
    
    # original surface height :
    self.origZ = zs

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

    zmax   = zs + 20                             # max z-coord
    zmin   = zb                                  # min z-coord

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
    self.phT,     = self.Tax.plot(self.T - 273.15, z, 'r-')
    self.phTs,    = self.Tax.plot([Tmin, Tmax], [zs, zs], 'k-', lw=2)
    self.phTs_0,  = self.Tax.plot(Th, self.origZ, 'ko')
    self.phTsp,   = self.Tax.plot(Th*np.ones(len(self.z)), self.z, 'r+')

    self.phrho,   = self.rhoax.plot(self.rho, self.z, 'g-')
    self.phrhoS,  = self.rhoax.plot([rhoMin, rhoMax], [zs, zs], 'k-', lw=2)
    self.phrhoS_0,= self.rhoax.plot(rhoh, self.origZ, 'ko')
    self.phrhoSp, = self.rhoax.plot(rhoh*np.ones(len(self.z)), self.z, 'r+')

    self.phw,     = self.wax.plot(self.w, self.z, 'b-')
    self.phws,    = self.wax.plot([wMin, wMax], [zs, zs], 'k-', lw=2)
    self.phws_0,  = self.wax.plot(wh, self.origZ, 'ko')
    self.phwsp,   = self.wax.plot(wh*np.ones(len(self.z)), self.z, 'r+')

    self.phk1,    = self.kax.plot(self.k1, self.z, label='Van Dusen')
    self.phk2,    = self.kax.plot(self.k2, self.z, label='Arthern')
    self.phk3,    = self.kax.plot(self.k3, self.z, label='Schwerdtfeger' )
    self.phks,    = self.kax.plot([kMin, kMax], [zs, zs], 'k-', lw=2)
    self.phks_0,  = self.kax.plot(kh, self.origZ, 'ko')
    self.phksp,   = self.kax.plot(kh*np.ones(len(self.z)), self.z, 'r+')

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


  def update_plot(self, T, rho, w, k1, k2, k3, z, origZ, t):
    
    self.T     = T
    self.rho   = rho
    self.w     = w
    self.k1    = k1
    self.k2    = k2
    self.k3    = k3
    self.z     = z
    self.origZ = origZ

    self.fig_text.set_text('Time = %.2f yr' % t) 
    self.phT.set_xdata(T[self.index] - 273.15)
    self.phT.set_ydata(z)
    self.phTs.set_ydata(z[-1])
    self.phTs_0.set_ydata(origZ)
    self.phTsp.set_ydata(z)
    
    self.phrho.set_xdata(rho[self.index])
    self.phrho.set_ydata(z)
    self.phrhoS.set_ydata(z[-1])
    self.phrhoS_0.set_ydata(origZ)
    self.phrhoSp.set_ydata(z)
   
    self.phw.set_xdata(w[self.index])
    self.phw.set_ydata(z)
    self.phws.set_ydata(z[-1])
    self.phws_0.set_ydata(origZ)
    self.phwsp.set_ydata(z)
    
    self.phk1.set_xdata(k1[self.index])
    self.phk2.set_xdata(k2[self.index])
    self.phk3.set_xdata(k3[self.index])
    self.phk1.set_ydata(z)
    self.phk2.set_ydata(z)
    self.phk3.set_ydata(z)
    self.phks.set_ydata(z[-1])
    self.phks_0.set_ydata(origZ)
    self.phksp.set_ydata(z)


  def plot_height(x, ht, origHt):

    # plot the surface height information :
    plt.plot(x, ht,     label='Surface Height')
    plt.plot(x, origHt, label='Original Surface')
    plt.xlabel(r'time $[a]$')
    plt.ylabel(r'height $[m]$')
    plt.title('Surface Height Changes')
    plt.grid()
  
    # Legend formatting:
    leg = plt.legend(loc='upper right')
    ltext  = leg.get_texts()
    frame  = leg.get_frame()
    plt.setp(ltext, fontsize='small')
    frame.set_alpha(0)
    plt.show()




