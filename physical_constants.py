"""
This module contains the classes that hold known 
physical constants relavent to the simulations and a class to create new 
constants
"""

class PhysicalConstant(float):
  """
  This class allows the creation of new floating point physical constants.
      
  :param float value: Value of the physical constant
  :param description: Description of the physical constant
  :param units: Units of the physical constant
  """
  def __new__(cls, value = 0.0, description = None, units = None):
    """
    Creates a new PhysicalConstant object
    """
    ii = float.__new__(cls,value)
    ii.description = description
    ii.units = units
    return ii

class FirnParameters(object):
  """
  This class contains the default physical parameters used in modeling
  the ice sheet.
  
  :param params: Optional dictionary object of physical parameters
  """
  def __init__(self,params=None):
    if params:
      self.params = params
    else:
      self.params = self.get_default_parameters()

  def globalize_parameters(self, namespace=None):
    """
    This function converts the parameter dictinary into global PhysicalContstant
    objects
    
    :param namespace: Optional namespace in which to place the global variables
    """
    for param in self.params.iteritems():
      vars(namespace)[param[0]] = PhysicalConstant(param[1][0],
                                                   param[1][1],
                                                   param[1][2])

  def get_default_parameters(self):
    """
    Creates a dictionary of default physical constants and returns it
    
    :rtype: Python dictionary
    """
    cpi   = 2009.
    Tw    = 273.15
    Hsp   = cpi * Tw

    d_params = \
    {'g'     : (9.81,        'gravitational acceleration',    'm/s^2'),
     'R'     : (8.3144621,   'gas constant',                  'J/(mol K)'),
     'spy'   : (31556926.0,  'seconds per year',              's/a'),
     'rhoi'  : (917.,        'density of ice',                'kg/m^3'),
     'rhoin' : (917.,        'initial density of column',     'kg/m^3'),
     'rhow'  : (1000.,       'density of water',              'kg/m^3'),
     'rhom'  : (550.,        'density at 15 m',               'kg/m^3'),
     'rhoc'  : (815.,        'density at critical value',     'kg/m^3'),
     'ki'    : (2.1,         'thermal conductivity of ice',   'W/(m K)'),
     'cpi'   : (2009.,       'const. heat capacitity of ice', 'J/(kg K)'),
     'kcHh'  : (3.7e-9,      'creep coefficient high',        '(m^3 s)/kg'),
     'kcLw'  : (9.2e-9,      'creep coefficient low ',        '(m^3 s)/kg'),
     'kg'    : (1.3e-7,      'grain growth coefficient',      'm^2/s'),
     'Ec'    : (60e3,        'act. energy for water in ice',  'J/mol'),
     'Eg'    : (42.4e3,      'act. energy for grain growth',  'J/mol'),
     'Tw'    : (273.15,      'triple point water',            'degrees K'),
     'T0'    : (0.0,         'reference temperature',         'K'),
     'beta'  : (7.9e-8,      'Clausius-Clapeyron',            'K/Pa'),
     'Lf'    : (3.34e5,      'latent heat of fusion',         'J/kg'),
     'Hsp'   : (Hsp,         'Enthalpy of ice at Tw',         'J/kg')}

    return d_params








  
