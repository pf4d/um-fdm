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
from fenics import *

class Enthalpy(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot

    dh      = firn.dh                        # trial function for solution
    psi     = firn.psi                       # test function for H
    phi     = firn.phi                       # test function for rho
    eta     = firn.eta                       # test function for w
  
    A       = firn.A
    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
    Hsp     = firn.Hsp
    Tw      = firn.Tw
    
    h       = firn.h                         # enthalpy, density, velocity
    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rho_1   = firn.rho_1                     # previous density
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    h_1     = firn.h_1                       # previous step's solution
    k       = firn.k                         # thermal conductivity
    c       = firn.c                         # heat capacity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    Tcoef   = firn.Tcoef                     # T above Tw = 0.0 coefficient
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    rhoCoef = firn.rhoCoef                   # density ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt                        # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    Ta      = interpolate(Constant(Tavg), V)
    c       = interpolate(Constant(cpi), V)
    k       = 2.1*(rho / rhoi)**2                          # Arthern 2008
    #Tcoef   = conditional( lt(T, Tw), 1.0, c / H * Tw )
    T       = Tcoef * H / c                                # temperature

    # enthalpy residual :
    theta     = 0.5
    H_mid     = theta*H + (1 - theta)*H_1
    Kcoef     = conditional( lt(H, Hsp), 1.0, 1.0/10.0 )
    f_H       = - k/(rho*c) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
                + (w-m) * H_mid.dx(0) * psi * dx \
                - (H - H_1)/dt * psi * dx
    
    # density residual :
    # material derivative :
    #  dr   pr     pr
    #  -- = -- + w --
    #  dt   pt     pz
    # SUPG method phihat :
    #rhoCoef = conditional( gt(rho, rhom), 
    #                       kcHh * (2.366 - 0.293*ln(A)),
    #                       kcLw * (1.435 - 0.151*ln(A)) )
    
    vnorm     = sqrt(dot(w-m, w-m) + 1e-10)
    cellh     = CellSize(mesh)
    phihat    = phi + cellh/(2*vnorm)*dot(w-m, phi.dx(0))
    
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * \
                (rhoi - rho_mid)
    f_rho     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phihat * dx \
                + (w-m) * rho_mid.dx(0) * phihat * dx 
    
    # velocity residual :
    theta     = 0.878
    w_mid     = theta*w + (1 - theta)*w_1
    # Zwally equation for surface velocity :
    f_w       = + rho * w_mid.dx(0) * eta * dx \
                + drhodt * eta * dx
    # Arthern equation of strain rate from 'Sorge's Law' :
    #f_w       = + rho**2 * w_mid.dx(0) * eta * dx \
    #            - bdot * rho.dx(0) * eta * dx
    
    # equation to be minimzed :
    f         = f_H + f_rho + f_w
    J         = derivative(f, h, dh)   # temp/density jacobian

    self.f = f
    self.J = J

  def solve(self):
    """
    """
    firn   = self.firn
    config = self.config

    h     = firn.h
    bcs   = [firn.HBc, firn.rhoBc, firn.wBc]

    # newton's iterative method :
    #epi = np.random.rand(self.firn.n)
    #h.vector().set_local(h.vector().array() + epi)
    solve(self.f == 0, h, bcs, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    
    firn.H, firn.rho, firn.w = h.split(True)


class Enthalpy_n(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot

    dh      = firn.dh                        # trial function for solution
    psi     = firn.psi                       # test function for H
    phi     = firn.phi                       # test function for rho
    eta     = firn.eta                       # test function for w
  
    A       = firn.A
    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
    Hsp     = firn.Hsp
    Tw      = firn.Tw
    
    h       = firn.h                         # enthalpy, density, velocity
    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rho_1   = firn.rho_1                     # previous density
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    h_1     = firn.h_1                       # previous step's solution
    k       = firn.k                         # thermal conductivity
    c       = firn.c                         # heat capacity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    Tcoef   = firn.Tcoef                     # T above Tw = 0.0 coefficient
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt                        # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    Ta      = interpolate(Constant(Tavg), V)
    c       = interpolate(Constant(cpi), V)
    k       = 2.1*(rho / rhoi)**2                          # Arthern 2008
    #Tcoef   = conditional( lt(T, Tw), 1.0, c / H * Tw )
    T       = Tcoef * H / c                                # temperature

    # enthalpy residual :
    theta     = 0.5
    H_mid     = theta*H + (1 - theta)*H_1
    Kcoef     = conditional( lt(H, Hsp), 1.0, 1.0/10.0 )
    delta     = - k/(rho*c) * Kcoef * inner(H_mid.dx(0), psi.dx(0)) * dx \
                + (w-m) * H_mid.dx(0) * psi * dx \
                - (H - H_1)/dt * psi * dx
    
    # equation to be minimzed :
    J         = derivative(delta, H, dH)   # temp/density jacobian

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    #epi = np.random.rand(self.firn.n)
    #h.vector().set_local(h.vector().array() + epi)
    solve(self.delta == 0, firn.H, firn.Hbc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])


class Density(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot

    dh      = firn.dh                        # trial function for solution
    psi     = firn.psi                       # test function for H
    phi     = firn.phi                       # test function for rho
    eta     = firn.eta                       # test function for w
  
    A       = firn.A
    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
    Hsp     = firn.Hsp
    Tw      = firn.Tw
    
    h       = firn.h                         # enthalpy, density, velocity
    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rho_1   = firn.rho_1                     # previous density
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    h_1     = firn.h_1                       # previous step's solution
    k       = firn.k                         # thermal conductivity
    c       = firn.c                         # heat capacity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    Tcoef   = firn.Tcoef                     # T above Tw = 0.0 coefficient
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    rhoCoef = firn.rhoCoef                   # density ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt                        # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    Ta      = interpolate(Constant(Tavg), V)
    c       = interpolate(Constant(cpi), V)
    k       = 2.1*(rho / rhoi)**2            # Arthern 2008
    T       = firn.T                         # temperature
    
    # material derivative :
    #  dr   pr     pr
    #  -- = -- + w --
    #  dt   pt     pz
    # SUPG method phihat :
    #rhoCoef = conditional( gt(rho, rhom), 
    #                       kcHh * (2.366 - 0.293*ln(A)),
    #                       kcLw * (1.435 - 0.151*ln(A)) )
    
    vnorm     = sqrt(dot(w-m, w-m) + 1e-10)
    cellh     = CellSize(mesh)
    phihat    = phi + cellh/(2*vnorm)*dot(w-m, phi.dx(0))
    
    theta     = 0.878
    rho_mid   = theta*rho + (1 - theta)*rho_1
    
    drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * \
                (rhoi - rho_mid)
    delta     = + (rho - rho_1)/dt * phi * dx \
                - drhodt * phihat * dx \
                + (w-m) * rho_mid.dx(0) * phihat * dx 
    
    J         = derivative(delta, rho, drho)

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    solve(self.delta == 0, firn.rho, firn.rhoBcs, J=self.J, 
          solver_parameters=config['density']['solver_params'])


class Velocity(object):

  def __init__(self, firn, config):
    """
    """
    self.firn   = firn
    self.config = config

    mesh    = firn.mesh
    V       = firn.V
    spy     = firn.spy
    cpi     = firn.cpi
    adot    = firn.adot

    dh      = firn.dh                        # trial function for solution
    psi     = firn.psi                       # test function for H
    phi     = firn.phi                       # test function for rho
    eta     = firn.eta                       # test function for w
  
    A       = firn.A
    kcHh    = firn.kcHh
    kcLw    = firn.kcLw
    Hsp     = firn.Hsp
    Tw      = firn.Tw
    
    h       = firn.h                         # enthalpy, density, velocity
    H       = firn.H                         # enthalpy
    H_1     = firn.H_1                       # previous enthalpy
    T       = firn.T                         # temperature
    rho     = firn.rho                       # density
    rho_1   = firn.rho_1                     # previous density
    w       = firn.w                         # velocity
    w_1     = firn.w_1                       # previous step's velocity
    m       = firn.m                         # mesh velocity
    h_1     = firn.h_1                       # previous step's solution
    k       = firn.k                         # thermal conductivity
    c       = firn.c                         # heat capacity
    bdot    = firn.bdot                      # average annual accumulation
    Tavg    = firn.Tavg                      # average surface temperature
    Tcoef   = firn.Tcoef                     # T above Tw = 0.0 coefficient
    Kcoef   = firn.Kcoef                     # enthalpy ceofficient
    rhoCoef = firn.rhoCoef                   # density ceofficient
    Ta      = firn.Ta                        # average temperature 
    dt      = firn.dt                        # timestep
    g       = firn.g                         # gravitational acceleration
    kg      = firn.kg                        # grain growth coefficient
    Ec      = firn.Ec                        # act. energy for water in ice
    Eg      = firn.Eg                        # act. energy for grain growth
    R       = firn.R                         # universal gas constant
    rhoi    = firn.rhoi                      # density of ice
    rhom    = firn.rhom                      # critical density
    
    bdot    = interpolate(Constant(rhoi * adot / spy), V)  # average annual acc
    #c       = (152.5 + sqrt(152.5**2 + 4*7.122*H)) / 2    # Patterson 1994
    Ta      = interpolate(Constant(Tavg), V)
    c       = interpolate(Constant(cpi), V)
    k       = 2.1*(rho / rhoi)**2                          # Arthern 2008
    #Tcoef   = conditional( lt(T, Tw), 1.0, c / H * Tw )
    T       = Tcoef * H / c                                # temperature

    # velocity residual :
    theta     = 0.878
    w_mid     = theta*w + (1 - theta)*w_1
    # Zwally equation for surface velocity :
    drhodt    = bdot*g*rhoCoef/kg * exp( -Ec/(R*T) + Eg/(R*Ta) ) * \
                (rhoi - rho_mid)
    delta     = + rho * w_mid.dx(0) * eta * dx \
                + drhodt * eta * dx
    # Arthern equation of strain rate from 'Sorge's Law' :
    #f_w       = + rho**2 * w_mid.dx(0) * eta * dx \
    #            - bdot * rho.dx(0) * eta * dx
    
    # equation to be minimzed :
    J         = derivative(delta, w, dw)   # temp/density jacobian

    self.delta = delta
    self.J     = J

  def solve(self):
    """
    """
    firn   = self.firn
    config = self.config

    # newton's iterative method :
    #epi = np.random.rand(self.firn.n)
    #h.vector().set_local(h.vector().array() + epi)
    solve(self.delta == 0, h, firn.wBc, J=self.J, 
          solver_parameters=config['enthalpy']['solver_params'])
    

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
    dt      = firn.dt                        # timestep
    
    # age residual :
    # theta scheme (1=Backwards-Euler, 0.667=Galerkin, 0.878=Liniger, 
    #               0.5=Crank-Nicolson, 0=Forward-Euler) :
    # uses Taylor-Galerkin upwinding :
    theta     = 0.5 
    a_mid     = theta*a + (1-theta)*a_1
    f         = + (a - a_1)/dt * xi * dx \
                - 1 * xi * dx \
                + (w-m) * a_mid.dx(0) * xi * dx \
                - 0.5 * ((w-m) - (w_1-m_1)) * a_mid.dx(0) * xi * dx \
                + (w-m)**2 * dt/2 * inner(a_mid.dx(0), xi.dx(0)) * dx \
                - (w-m) * (w-m).dx(0) * dt/2 * a_mid.dx(0) * xi * dx
    J         = derivative(f, a, da) # age jacobian

    self.f = f
    self.J = J


  def solve(self):
    """
    """
    a      = self.firn.a
    ageBc  = self.firn.ageBc
    config = self.config

    # solve for age :
    solve(self.f == 0, a, ageBc, J=self.J,
          solver_parameters=config['age']['solver_params'])
  
