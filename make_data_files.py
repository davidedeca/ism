import os 
import BEsphere.BesphereDB as bdb

#download Gnedin & Hollon table

#download leiden and swri cross sections for krome 

#run BE save table

print("Creating database for BE spheres")
bdb.save_BEtable(1000000, 1000.)
print(".. done")
