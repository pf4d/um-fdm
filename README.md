um-fdm
======

University Of Montana Firn Densification Model


Evan Cummings
07.12.12

FEniCS solution to firn temperature/density profile.

run with "python objModel.py <model> <end time> <initialize>

where model is:
  zl ... Li and Zwally 2002 model with Reeh correction.
  hl ... Herron and Langway 1980 [unworking]
  a .... Arthern 2008

end time is time to run the model in years, and

initialize is either 'i' or any other character:
  'i' initializes the temperature and density profile to a 180-year converge.
