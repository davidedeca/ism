import ism.pykrome2.pykrome as pk
from ism.utils.constants import *

gas = pk.cell(1.e3, 100., crate=0.)
gas.evolution(Myr)
