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
from pylab     import intersect1d, where, ones
from termcolor import colored


class Enthalpy(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    Q       = firn.Q

    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    w       = firn.w                         # velocity
    m       = firn.m                         # mesh velocity
    Tavg    = firn.Tavg                      # average surface temperature
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    dt      = firn.dt_v                      # timestep
    rhoi    = firn.rhoi                      # density of ice
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot
    bdot    = firn.bdot
    T       = firn.T
    Tw      = firn.Tw
    T0      = firn.T0
    Lf      = firn.Lf
    Hsp     = firn.Hsp
    u       = firn.u
    p       = firn.p
    ql      = firn.ql
    etaw    = firn.etaw
    rhow    = firn.rhow
    #w       = w - m
    z       = firn.x
    g       = firn.g
    r       = firn.r
    S       = firn.S
    #omega   = firn.omega
    
    psi   = TestFunction(Q)
    dH    = TrialFunction(Q)
    
    # thermal parameters
    ki  = 2.1*(rho / rhoi)**2
    ci  = cpi#152.5 + 7.122*T
   
    # Darcy flux :
    omega = conditional(lt(H, Hsp), 0, (H - ci*(Tw - T0))/Lf)
    k     = 0.077 * r * exp(-7.8*rho/rhow)                # intrinsic perm.
    phi   = 1 - rho/rhoi                                  # porosity
    Smi   = 0.0057 / (1 - phi) + 0.017                    # irr. water content
    Se    = (omega - Smi) / (1 - Smi)                     # effective sat.
    K     = k * rhow * g / etaw
    krw   = Se**3.0 
    ql    = K * krw * (p / (rhow * g) + z).dx(0)          # water flux
    u     = - k / etaw * p.dx(0)                          # darcy velocity
    u     = - ql/phi                                      # darcy velocity

    # enthalpy residual :
    theta   = 1.0
    H_mid   = theta*H + (1 - theta)*H_1
    delta   = - ki/(rho*ci) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
              + w * H_mid.dx(0) * psi * dx \
              + (ql * H_mid).dx(0) * psi * dx \
              - (H - H_1)/dt * psi * dx
    
    # equation to be minimzed :
    J       = derivative(delta, H, dH)   # temp/density jacobian

    self.delta = delta
    self.J     = J
    self.u     = u
    self.ql    = ql
    self.omega = omega
    self.Smi   = Smi

  def solve(self):
    """
    """
    s    = "::: solving enthalpy :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    solve(self.delta == 0, firn.H, firn.HBc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])

    firn.omega = project(self.omega)
    firn.ql    = project(self.ql)
    
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
    g     = firn.g

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
    domega           = omegap - omegap_1    # water content chg.

    # update the dolfin vectors :
    firn.assign_variable(firn.T,       Tp)
    firn.assign_variable(firn.omega_1, firn.omega)
    #firn.assign_variable(firn.Kcoef,   KcoefNew)
    
    firn.domega = domega

    firn.print_min_max(firn.T,     'T')
    firn.print_min_max(firn.H,     'H')
    firn.print_min_max(firn.omega, 'omega')
    firn.print_min_max(firn.ql,    'ql')

    p = firn.vert_integrate(rhow * g * firn.omega)
    rho   = firn.rho
    phi   = 1 - rho/rhoi                                    # porosity
    Smi   = 0.0057 / (1 - phi) + 0.017                      # irr. water content
    firn.assign_variable(firn.p,   p)
    firn.assign_variable(firn.u,   project(self.u))
    firn.assign_variable(firn.Smi, project(Smi))
    firn.print_min_max(firn.pp, 'p')
    firn.print_min_max(firn.up, 'u')
    firn.print_min_max((1-firn.rhop/rhoi)*100 - firn.omegap*100, 'phi - omega')
    

class Density(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    Q       = firn.Q

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
    dt      = firn.dt_v                      # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    T       = firn.T                         # temperature
    
    phi     = TestFunction(Q)
    drho    = TrialFunction(Q)
  
    # SUPG method phihat :
    vnorm     = sqrt(dot(w, w) + DOLFIN_EPS)
    cellh     = CellSize(mesh)
    phihat    = phi + cellh/(2*vnorm)*dot(w, phi.dx(0))
    
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*T) ) * \
                (rhoi - rho_mid)
    delta     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phi * dx \
                + w * rho_mid.dx(0) * phi * dx 
    
    J         = derivative(delta, rho, drho)

    self.delta  = delta
    self.J      = J
    firn.drhodt = drhodt


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
    A                   = firn.rhoi/firn.rhow * 1e3 * firn.adot
    rhoCoefNew          = ones(firn.n)
    rhoHigh             = where(rhop >  550)[0]
    rhoLow              = where(rhop <= 550)[0]
    rhoCoefNew[rhoHigh] = firn.kcHh * (2.366 - 0.293*ln(A))
    rhoCoefNew[rhoLow]  = firn.kcLw * (1.435 - 0.151*ln(A))
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

    #firn.assign_variable(firn.rho, rhop)
    firn.print_min_max(firn.rho, 'rho')
  

class FullDensity(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    Q       = firn.Q

    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
   
    H       = firn.H                         # enthalpy
    T       = firn.T                         # temperature
    w       = firn.w                         # velocity
    m       = firn.m                         # mesh velocity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    rhoCoef = firn.rhoCoef                   # density ceofficient
    dt      = firn.dt_v                      # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    T       = firn.T                         # temperature
  
    Q       = MixedFunctionSpace([Q,Q,Q])
    dQ      = TrialFunction(Q)

    U       = Function(Q)
    U_1     = Function(Q)
    Phi     = TestFunction(Q)

    rho,   sigma,   r    = U
    rho_1, sigma_1, r_1  = U_1
    phi,   psi,     xi   = Phi

    # initialize :
    U_i = project(as_vector([firn.rho_i, firn.sigma_i, firn.r_i]), Q)
    firn.assign_variable(U,   U_i)
    firn.assign_variable(U_1, U_i)
  
    # rho residual :
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = rhoCoef * exp( -Ec/(R*T) ) * (rhoi - rho_mid) * sigma / r
    d_rho     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phi * dx \
                + w * rho_mid.dx(0) * phi * dx 
    
    # sigma residual : 
    theta     = 0.878
    sig_mid   = theta*sigma + (1 - theta)*sigma_1
    dsigdt    = bdot * g
    d_sigma   = + (sigma - sigma_1)/dt * psi * dx \
                - dsigdt * psi * dx \
                + w * sig_mid.dx(0) * psi * dx

    # r residual :
    theta   = 0.878
    r_mid   = theta*r + (1 - theta)*r_1
    drdt    = kg * exp( -Eg/(R*T) )
    d_r     = + (r - r_1)/dt * xi * dx \
              - drdt * xi * dx \
              + w * r_mid.dx(0) * xi * dx

    surface = firn.surface
    rhoBc   = DirichletBC(Q.sub(0), firn.rho_S,   surface)
    sigmaBc = DirichletBC(Q.sub(1), firn.sigma_S, surface)
    rBc     = DirichletBC(Q.sub(2), firn.r_S,     surface)

    self.bcs = [rhoBc, sigmaBc, rBc]
    
    self.delta  = d_rho + d_sigma + d_r
    self.J      = derivative(self.delta, U, dQ)
    firn.rho    = rho
    firn.sigma  = sigma
    firn.r      = r
    firn.U      = U
    firn.U_1    = U_1
    self.drhodt = drhodt

  def solve(self):
    """
    """
    s    = "::: solving density, overburden stress, and grain radius :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    solve(self.delta == 0, firn.U, bcs=self.bcs, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    firn.rho, firn.sigma, firn.r = firn.U.split(True)

    rhop = firn.rho.vector().array()

    # update kc term in drhodt :
    # if rho >  550, kc = kcHigh
    # if rho <= 550, kc = kcLow
    # with parameterizations given by ligtenberg et all 2011
    A                   = firn.rhoi/firn.rhow * 1e3 * firn.adot
    rhoCoefNew          = ones(firn.n)
    rhoHigh             = where(rhop >  550)[0]
    rhoLow              = where(rhop <= 550)[0]
    rhoCoefNew[rhoHigh] = firn.kcHh * (2.366 - 0.293*ln(A))
    rhoCoefNew[rhoLow]  = firn.kcLw * (1.435 - 0.151*ln(A))
    firn.assign_variable(firn.rhoCoef, rhoCoefNew)
    firn.assign_variable(firn.drhodt,  project(self.drhodt))
    
    #firn.assign_variable(firn.rho, rhop)
    firn.print_min_max(firn.rho,   'rho')
    firn.print_min_max(firn.sigma, 'sigma')
    firn.print_min_max(firn.r,     'r^2')


class Velocity(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    Q       = firn.Q

    rho     = firn.rho                       # density
    w       = TrialFunction(Q)               # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    bdot    = firn.bdot                      # average annual accumulation
    dt      = firn.dt_v                      # timestep
    drhodt  = firn.drhodt
    
    eta     = TestFunction(Q)
    dw      = TrialFunction(Q)

    # velocity residual :
    theta   = 0.878
    w_mid   = theta*w + (1 - theta)*w_1
    
    delta   = + rho * w_mid.dx(0) * eta * dx \
              + drhodt * eta * dx
    
    ## Arthern equation of strain rate from 'Sorge's Law' :
    #delta   = + rho**2 * w_mid.dx(0) * eta * dx \
    #          - bdot * rho.dx(0) * eta * dx
    
    self.delta = delta

  def solve(self):
    """
    """
    s    = "::: solving velocity :::"
    text = colored(s, 'cyan')
    print text
    
    firn   = self.firn
    config = self.config
    delta  = self.delta

    # linear solve :
    solve(lhs(delta) == rhs(delta), firn.w, firn.wBc)
    firn.print_min_max(firn.w, 'w')


class Age(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    Q       = firn.Q
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    m_1     = firn.m_1                       # previous mesh velocity
    a       = firn.a                         # age
    a_1     = firn.a_1                       # previous step's age
    dt      = firn.dt_v                      # timestep
    
    da      = TrialFunction(Q)
    xi      = TestFunction(Q)

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
  

