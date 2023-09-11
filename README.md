# Tools for ISM Physics

A set of tools for various typical calculation in the Physics of the Interstellar Medium:
- **BEsphere**: compute properties for a Bonnor-Ebert sphere;
- **HII_HI_regions**: compute the ionization and atomic fraction profile in a HII region and a Photodissociation region;
- **photoevaporation**: compute the photoevaporation time and the radius as a function of time during the photoevaporation of a molecular clump invested by radiation;
- **pykrome**: wrapper for the pykrome python library for krome;
- **ramses_utils**: tools to analyse ramses snapshots;
- **sph_utils**: tools to analyse simulation snapshots with pynbody;
- **star_utils**: functions to compute quantities related to radiation emitted by different sources (stars, quasars)
- **utils**: various useful tools;
- **wave_propagation**: compute hydro quantities for shock and rarefaction waves. 

# Installation

Clone the repo and add it to the ```PYTHONPATH``` in your ```.bashrc```.
Some other common packages for Astrophysics are needed: pymses, pynbody, astropy.

Some fortran libraries need to be compiled locally on your machine. In particular:

For pykrome:
```bash
cd pykrome
make gfortran
```

For Gnedin-Hollon cooling functions:
```bash
cd GnedinCooling
f2py -c phfit2.f -m cross_section
f2py -c frt_cf3m.F -m gnedinhollon
```
