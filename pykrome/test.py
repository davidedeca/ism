import utils.krome_singlezone as ks
from utils.constants import *

x = ks.cell(100., 50., zred=3.7)
x.evolution(10*Myr, Myr)
