from pylab          import plt, linspace
from physics        import Enthalpy, Density, Velocity, Age
from plot           import Plot
from termcolor      import colored, cprint


class TransientSolver(object):
  """
  """
  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    # form the physics :
    self.fe = Enthalpy(firn, config)
    self.fv = Velocity(firn, config)
    self.fd = Density(firn, config)
    self.fa = Age(firn, config)

    if config['plot']:
      plt.ion() 
      self.plot = Plot(firn)
      plt.draw()

  def solve(self):
    """
    """
    firn   = self.firn
    config = self.config

    fe     = self.fe
    fv     = self.fv
    fd     = self.fd
    fa     = self.fa
    
    t0    = config['t_start']
    tf    = config['t_end']
    dt    = config['time_step']
    numt  = (tf-t0)/dt + 1         # number of time steps
    times = linspace(t0,tf,numt)   # array of times to evaluate in seconds
    self.times = times

    for t in times[1:]:
      # update boundary conditions :
      firn.update_Hbc()
      #firn.update_rhoBc()
    
      # newton's iterative method :
      fe.solve()
      fv.solve()
      fd.solve()
      fa.solve()
      
      # update firn object :
      firn.update_vars(t)
      firn.update_height_history()
      firn.update_height()
      
      # update model parameters :
      if t != times[-1]:
         firn.H_1.assign(firn.H)
         firn.rho_1.assign(firn.rho)
         firn.omega_1.assign(firn.omega)
         firn.w_1.assign(firn.w)
         firn.a_1.assign(firn.a)
         firn.m_1.assign(firn.m)
    
      # update the plotting parameters :
      if config['plot']:
        self.plot.update_plot()
        plt.draw()
    
    if config['plot']:
      plt.ioff()
      plt.show()



