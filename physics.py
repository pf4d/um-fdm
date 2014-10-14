#    Copyright (C) <2012>  <cummings.evan@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
model.py
Evan Cummings
01.16.12

FEniCS solution to firn enthalpy / density profile.

"""
from fenics    import *
from pylab     import intersect1d, where, zeros, ones
from termcolor import colored

class Enthalpy(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V

    psi     = firn.psi                       # test function for H
  
    dH      = firn.dH
    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    w       = firn.w                         # velocity
    m       = firn.m                         # mesh velocity
    k       = firn.k                         # thermal conductivity
    c       = firn.c                         # heat capacity
    Tavg    = firn.Tavg                      # average surface temperature
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt_v                      # timestep
    rhoi    = firn.rhoi                      # density of ice
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot
    bdot    = firn.bdot
    Ta      = firn.Ta

    w       = w - m

    # SUPG method psihat :
    vnorm   = sqrt(dot(w, w) + 1e-10)
    cellh   = CellSize(mesh)
    psihat  = psi + cellh/(2*vnorm)*dot(w, psi.dx(0))

    # enthalpy residual :
    theta   = 0.5
    H_mid   = theta*H + (1 - theta)*H_1
    delta   = - k/(rho*c) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
              + w * H_mid.dx(0) * psihat * dx \
              - (H - H_1)/dt * psi * dx
    
    # equation to be minimzed :
    J       = derivative(delta, H, dH)   # temp/density jacobian

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    s    = "::: solving enthalpy :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    #epi = np.random.rand(self.firn.n)
    #h.vector().set_local(h.vector().array() + epi)
    solve(self.delta == 0, firn.H, firn.HBc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    
    n     = firn.n
    Tw    = firn.Tw
    T0    = firn.T0
    Lf    = firn.Lf
    rhow  = firn.rhow
    rhoi  = firn.rhoi
    kcHh  = firn.kcHh
    kcLw  = firn.kcLw
    Hsp   = firn.Hsp
    index = firn.index

    # find vector of T, rho :
    Hp       = firn.H.vector().array()
    Tp       = Hp / firn.cp
    omegap   = firn.omega.vector().array()
    omegap_1 = firn.omega_1.vector().array()

    # update coefficients used by enthalpy :
    Hhigh            = where(Hp > Hsp)[0]
    Hlow             = where(Hp < Hsp)[0]
    KcoefNew         = ones(n)
  
    KcoefNew[Hhigh]  = 1.0/2.0
    KcoefNew[Hlow]   = 1.0
    Tp[Hhigh]        = Tw
  
    # update water content and density :
    omegap[Hhigh]    = (Hp[Hhigh] - firn.cp[Hhigh]*(Tw - T0)) / Lf
    omegap[Hlow]     = 0.0
    domega           = omegap - omegap_1              # water content chg.

    # update the dolfin vectors :
    firn.assign_variable(firn.T,       Tp)
    firn.assign_variable(firn.omega_1, firn.omega)
    firn.assign_variable(firn.omega,   omegap)
    #firn.assign_variable(firn.Kcoef,   KcoefNew)
    
    firn.domega = domega
      
    firn.print_min_max(firn.T,     'T')
    firn.print_min_max(firn.H,     'H')
    firn.print_min_max(firn.omega, 'omega')


class Density(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V

    phi     = firn.phi                       # test function for rho
    drho    = firn.drho 
  
    A       = firn.A
    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
   
    H       = firn.H                         # enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rho_1   = firn.rho_1                     # previous density
    w       = firn.w                         # velocity
    m       = firn.m                         # mesh velocity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    rhoCoef = firn.rhoCoef                   # density ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt_v                      # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    c       = firn.c
    k       = firn.k
    Ta      = firn.Ta
    T       = firn.T                         # temperature

    w       = w - m

    # material derivative :
    #  dr   pr     pr
    #  -- = -- + w --
    #  dt   pt     pz
    #rhoCoef = conditional( gt(rho, rhom), 
    #                       kcHh * (2.366 - 0.293*ln(A)),
    #                       kcLw * (1.435 - 0.151*ln(A)) )
    
    # SUPG method phihat :
    vnorm     = sqrt(dot(w, w) + 1e-10)
    cellh     = CellSize(mesh)
    phihat    = phi + cellh/(2*vnorm)*dot(w, phi.dx(0))
    
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * \
                (rhoi - rho_mid)
    delta     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phi * dx \
                + w * rho_mid.dx(0) * phihat * dx 
    
    J         = derivative(delta, rho, drho)

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    s    = "::: solving density :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    solve(self.delta == 0, firn.rho, firn.rhoBc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    
    rhop = firn.rho.vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    rhoCoefNew          = ones(firn.n)
    rhoHigh             = where(rhop >  550)[0]
    rhoLow              = where(rhop <= 550)[0]
    rhoCoefNew[rhoHigh] = firn.kcHh * (2.366 - 0.293*ln(firn.A))
    rhoCoefNew[rhoLow]  = firn.kcLw * (1.435 - 0.151*ln(firn.A))
    firn.assign_variable(firn.rhoCoef, rhoCoefNew)
    
    rhow   = firn.rhow
    rhoi   = firn.rhoi
    domega = firn.domega

    # update density for water content :
    domPos       = where(domega > 0)[0]                # water content inc.
    domNeg       = where(domega < 0)[0]                # water content dec.
    rhoNotLiq    = where(rhop < rhow)[0]               # density < water
    rhoInc       = intersect1d(domPos, rhoNotLiq)      # where rho can inc.
    rhop[rhoInc] = rhop[rhoInc] + domega[rhoInc]*rhow 
    rhop[domNeg] = rhop[domNeg] + domega[domNeg]*(rhow - rhoi)

    firn.assign_variable(firn.rho, rhop)
    firn.print_min_max(firn.rho, 'rho')


class Velocity(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V

    eta     = firn.eta                       # test function for w
    dw      = firn.dw
  
    H       = firn.H                         # enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rhoi    = firn.rhoi
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    rhoCoef = firn.rhoCoef                   # density ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt_v                      # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    Ta      = firn.Ta

    # velocity residual :
    theta   = 0.878
    w_mid   = theta*w + (1 - theta)*w_1
    # Zwally equation for surface velocity :
    drhodt  = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * (rhoi - rho)
    delta   = + rho * w_mid.dx(0) * eta * dx \
              + drhodt * eta * dx
    # Arthern equation of strain rate from 'Sorge's Law' :
    #drhodt    = + rho**2 * w_mid.dx(0) * eta * dx \
    #            - bdot * rho.dx(0) * eta * dx
    
    # equation to be minimzed :
    J         = derivative(delta, w, dw)   # temp/density jacobian

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    s    = "::: solving velocity :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    #epi = np.random.rand(self.firn.n)
    #h.vector().set_local(h.vector().array() + epi)
    solve(self.delta == 0, firn.w, firn.wBc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    firn.print_min_max(firn.w, 'w')
    

class Age(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    da      = firn.da                        # trial function for age
    xi      = firn.xi                        # age test function
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    m_1     = firn.m_1                       # previous mesh velocity
    a       = firn.a                         # age
    a_1     = firn.a_1                       # previous step's age
    dt      = firn.dt_v                      # timestep

    w       = w - m
    w_1     = w_1 - m_1
    
    # age residual :
    # theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
    #               0.5=Crank-Nicolson, 0=Forward-Euler) :
    # uses Taylor-Galerkin upwinding :
    theta   = 0.5 
    a_mid   = theta*a + (1-theta)*a_1
    f       = + (a - a_1)/dt * xi * dx \
              - 1 * xi * dx \
              + w * a_mid.dx(0) * xi * dx \
              - 0.5 * (w - w_1) * a_mid.dx(0) * xi * dx \
              + w**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
              - w * w.dx(0) * dt/2 * a_mid.dx(0) * xi * dx
    J       = derivative(f, a, da) # age jacobian

    self.f = f
    self.J = J


  def solve(self):
    """
    """
    s    = "::: solving age :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    a      = firn.a
    ageBc  = firn.ageBc
    config = self.config

    # solve for age :
    solve(self.f == 0, a, ageBc, J=self.J,
          solver_parameters=config['age']['solver_params'])
    firn.print_min_max(firn.a, 'age')
  

