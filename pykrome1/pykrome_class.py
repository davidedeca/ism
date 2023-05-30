# -*- coding: utf-8 -*-
import numpy as np
import ctypes
import numpy.ctypeslib as npctypes

# define aliases for complicated variable types
int_byref = ctypes.POINTER(ctypes.c_int)
dble_byref = ctypes.POINTER(ctypes.c_double)
array_1d_int = npctypes.ndpointer(dtype=np.int,ndim=1,flags='CONTIGUOUS')
array_1d_double = npctypes.ndpointer(dtype=np.double,ndim=1,flags='CONTIGUOUS')
array_2d_double = npctypes.ndpointer(dtype=np.double,ndim=2,flags='CONTIGUOUS')


class PyKROME(object):
	# NOTE: Fortran is, by default, case-insensitive. As a side effect, all of the
	# symbols in `libkrome.so` are entirely lower-case (regardless of their form in
	# the Fortran files).

	def __init__(self, path_to_lib='.'):
		fortran = npctypes.load_library('libkrome.so',path_to_lib)

		# alias the Fortran functions and subroutines to `lib`.
		self.lib = fortran

		self.krome_idx_E =   0 #1 # E
		self.krome_idx_Hk =  1 #6 # H-
		self.krome_idx_H =   2 #0 # H
		self.krome_idx_HE =  3 #3 # HE
		self.krome_idx_H2 =  4 #7 # H2
		self.krome_idx_Hj =  5 #2 # H+
		self.krome_idx_HEj = 6 #4 # HE+
		self.krome_idx_H2j = 7 #8 # H2+
		self.krome_idx_HEjj =8 #5 # HE++
		self.krome_idx_CR = 9 # CR
		self.krome_idx_g = 10 # g
		self.krome_idx_Tgas = 11 # Tgas
		self.krome_idx_dummy = 12 # dummy
		self.krome_names = (
						"E",
						"H-",
						"H",
						"HE",
						"H2",
						"H+",
						"HE+",
						"H2+",
						"HE++",
						"CR",
						"g",
						"Tgas",
						"dummy"
						)

		self.krome_idx_cool_h2 = 0
		self.krome_idx_cool_h2gp = 1
		self.krome_idx_cool_atomic = 2
		self.krome_idx_cool_cen = 2
		self.krome_idx_cool_hd = 3
		self.krome_idx_cool_metal = 4
		self.krome_idx_cool_z = 4
		self.krome_idx_cool_dh = 5
		self.krome_idx_cool_enthalpic = 5
		self.krome_idx_cool_dust = 6
		self.krome_idx_cool_compton = 7
		self.krome_idx_cool_cie = 8
		self.krome_idx_cool_cont = 9
		self.krome_idx_cool_continuum = 9
		self.krome_idx_cool_expansion = 10
		self.krome_idx_cool_exp = 10
		self.krome_idx_cool_ff = 11
		self.krome_idx_cool_bss = 11
		self.krome_idx_cool_custom = 12
		self.krome_idx_cool_co = 13
		self.krome_idx_cool_zcie = 14
		self.krome_idx_cool_zcienouv = 15
		self.krome_idx_cool_zextend = 16
		self.krome_idx_cool_gh = 17
		self.krome_ncools = 18

		self.krome_idx_heat_chem = 0
		self.krome_idx_heat_compress = 1
		self.krome_idx_heat_compr = 1
		self.krome_idx_heat_photo = 2
		self.krome_idx_heat_dh = 3
		self.krome_idx_heat_enthalpic = 3
		self.krome_idx_heat_av = 4
		self.krome_idx_heat_photoav = 4
		self.krome_idx_heat_cr = 5
		self.krome_idx_heat_dust = 6
		self.krome_idx_heat_xray = 7
		self.krome_idx_heat_viscous = 8
		self.krome_idx_heat_visc = 8
		self.krome_idx_heat_custom = 9
		self.krome_idx_heat_zcie = 10
		self.krome_nheats = 11

		self.krome_nrea = 52
		self.krome_nmols = 9
		self.krome_nspec = 13
		self.krome_natoms = 3
		self.krome_ndust = 0
		self.krome_ndustTypes = 0
		self.krome_nPhotoBins = 10
		self.krome_nPhotoRates = 8

		self.krome_boltzmann_eV = 8.617332478e-5 # eV / K
		self.krome_boltzmann_J = 1.380648e-23 # J / K
		self.krome_boltzmann_erg = 1.380648e-16 # erg / K
		self.krome_iboltzmann_eV = 1e0/self.krome_boltzmann_eV # K / eV
		self.krome_iboltzmann_erg = 1e0/self.krome_boltzmann_erg # K / erg
		self.krome_planck_eV = 4.135667516e-15 # eV s
		self.krome_planck_J = 6.62606957e-34 # J s
		self.krome_planck_erg = 6.62606957e-27 # erg s
		self.krome_iplanck_eV = 1e0/self.krome_planck_eV # 1 / eV / s
		self.krome_iplanck_J = 1e0/self.krome_planck_J # 1 / J / s
		self.krome_iplanck_erg = 1e0/self.krome_planck_erg # 1 / erg / s
		self.krome_gravity = 6.674e-8 # cm3 / g / s2
		self.krome_e_mass = 9.10938188e-28 # g
		self.krome_p_mass = 1.67262158e-24 # g
		self.krome_n_mass = 1.674920e-24 # g
		self.krome_ip_mass = 1e0/self.krome_p_mass # 1/g
		self.krome_clight = 2.99792458e10 # cm/s
		self.krome_pi = 3.14159265359e0 # #
		self.krome_eV_to_erg = 1.60217646e-12 # eV -> erg
		self.krome_ry_to_eV = 13.60569e0 # rydberg -> eV
		self.krome_ry_to_erg = 2.179872e-11 # rydberg -> erg
		self.krome_seconds_per_year = 365e0*24e0*3600e0 # yr -> s
		self.krome_km_to_cm = 1e5 # km -> cm
		self.krome_cm_to_Mpc = 1.e0/3.08e24 # cm -> Mpc
		self.krome_kvgas_erg = 8.e0*self.krome_boltzmann_erg/self.krome_pi/self.krome_p_mass #
		self.krome_pre_kvgas_sqrt = np.sqrt(8.e0*self.krome_boltzmann_erg/self.krome_pi) #
		self.krome_pre_planck = 2.e0*self.krome_planck_erg/self.krome_clight**2 # erg/cm2*s3
		self.krome_exp_planck = self.krome_planck_erg / self.krome_boltzmann_erg # s*K
		self.krome_stefboltz_erg = 5.670373e-5 # erg/s/cm2/K4
		self.krome_N_avogadro = 6.0221e23 # #
		self.krome_Rgas_J = 8.3144621e0 # J/K/mol
		self.krome_Rgas_kJ = 8.3144621e-3 # kJ/K/mol
		self.krome_hubble = 0.704e0 # dimensionless
		self.krome_Omega0 = 1.0e0 # dimensionless
		self.krome_Omegab = 0.0456e0 # dimensionless
		self.krome_Hubble0 = 1.e2*self.krome_hubble*self.krome_km_to_cm*self.krome_cm_to_Mpc # 1/s

		fortran.krome_set_user_crate.restype = None
		fortran.krome_set_user_crate.argtypes = [ctypes.c_double]
		fortran.krome_get_user_crate.restype = ctypes.c_double
		fortran.krome_get_user_crate.argtypes = None
		fortran.krome_set_user_myfluxlw.restype = None
		fortran.krome_set_user_myfluxlw.argtypes = [ctypes.c_double]
		fortran.krome_get_user_myfluxlw.restype = ctypes.c_double
		fortran.krome_get_user_myfluxlw.argtypes = None
                fortran.krome_set_user_myh2_dissociation.restype = None
                fortran.krome_set_user_myh2_dissociation.argtypes = [ctypes.c_double]
                fortran.krome_get_user_myh2_dissociation.restype = ctypes.c_double
                fortran.krome_get_user_myh2_dissociation.argtypes = None
		fortran.krome_set_user_cell_size.restype = None
		fortran.krome_set_user_cell_size.argtypes = [ctypes.c_double]
		fortran.krome_get_user_cell_size.restype = ctypes.c_double
		fortran.krome_get_user_cell_size.argtypes = None

		fortran.krome_set_tcmb.restype = None
		fortran.krome_set_tcmb.argtypes = [ctypes.c_double]
		fortran.krome_get_tcmb.restype = ctypes.c_double
		fortran.krome_get_tcmb.argtypes = None
		fortran.krome_set_zredshift.restype = None
		fortran.krome_set_zredshift.argtypes = [ctypes.c_double]
		fortran.krome_get_zredshift.restype = ctypes.c_double
		fortran.krome_get_zredshift.argtypes = None
		fortran.krome_set_orthopararatio.restype = None
		fortran.krome_set_orthopararatio.argtypes = [ctypes.c_double]
		fortran.krome_get_orthopararatio.restype = ctypes.c_double
		fortran.krome_get_orthopararatio.argtypes = None
		fortran.krome_set_metallicity.restype = None
		fortran.krome_set_metallicity.argtypes = [ctypes.c_double]
		fortran.krome_get_metallicity.restype = ctypes.c_double
		fortran.krome_get_metallicity.argtypes = None
		fortran.krome_set_tfloor.restype = None
		fortran.krome_set_tfloor.argtypes = [ctypes.c_double]
		fortran.krome_get_tfloor.restype = ctypes.c_double
		fortran.krome_get_tfloor.argtypes = None


		# krome.f90 argument and return (result) types
		fortran.krome.restype = None
		fortran.krome_equilibrium.restype = None
		fortran.krome.argtypes = [array_1d_double, dble_byref, dble_byref]
		fortran.krome_equilibrium.argtypes = [array_1d_double, dble_byref]
		fortran.krome.argtypes.append(int_byref)
		fortran.krome_init.restype = None
		fortran.krome_init.argtypes = None
		fortran.krome_get_coe.restype = array_1d_double
		fortran.krome_get_coe.argtypes = [array_1d_double, dble_byref]
		fortran.krome_get_coet.restype = array_1d_double
		fortran.krome_get_coet.argtypes = [dble_byref]

		#krome_user.f90 argument and return (result) types
		fortran.krome_get_table_tdust.restype = None
		fortran.krome_get_table_tdust.argtypes = [array_1d_double, dble_byref]
		fortran.krome_num2col.restype = dble_byref
		fortran.krome_num2col.argtypes = [ctypes.c_double, array_1d_double, ctypes.c_double]
		fortran.krome_print_phys_variables.restype = None
		fortran.krome_print_phys_variables.argtypes = None
		fortran.krome_store.restype = None
		fortran.krome_store.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double]
		fortran.krome_restore.restype = None
		fortran.krome_restore.argtypes = [array_1d_double, dble_byref, dble_byref]
		fortran.krome_thermo_on.restype = None
		fortran.krome_thermo_on.argtypes = None
		fortran.krome_thermo_off.restype = None
		fortran.krome_thermo_off.argtypes = None
		fortran.krome_calc_photobins.restype = None
		fortran.krome_calc_photobins.argtypes = None
		fortran.krome_set_photobinj.restype = None
		fortran.krome_set_photobinj.argtypes = [array_1d_double]
		fortran.krome_set_photobine_lr.restype = None
		fortran.krome_set_photobine_lr.argtypes = [array_1d_double, array_1d_double]
		fortran.krome_set_photobine_moc.restype = None
		fortran.krome_set_photobine_moc.argtypes = [array_1d_double, array_1d_double]
		fortran.krome_set_photobine_lin.restype = None
		fortran.krome_set_photobine_lin.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobine_log.restype = None
		fortran.krome_set_photobine_log.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_get_photobinj.restype = array_1d_double
		fortran.krome_get_photobinj.argtypes = None
		fortran.krome_get_photobine_left.restype = array_1d_double
		fortran.krome_get_photobine_left.argtypes = None
		fortran.krome_get_photobine_right.restype = array_1d_double
		fortran.krome_get_photobine_right.argtypes = None
		fortran.krome_get_photobine_mid.restype = array_1d_double
		fortran.krome_get_photobine_mid.argtypes = None
		fortran.krome_get_photobine_delta.restype = array_1d_double
		fortran.krome_get_photobine_delta.argtypes = None
		fortran.krome_get_photobin_rates.restype = array_1d_double
		fortran.krome_get_photobin_rates.argtypes = None
		fortran.krome_get_xsec.restype = None
		fortran.krome_get_xsec.argtypes = [ctypes.c_int]
		fortran.krome_get_photobin_heats.restype = array_1d_double
		fortran.krome_get_photobin_heats.argtypes = None
		fortran.krome_photobin_scale.restype = None
		fortran.krome_photobin_scale.argtypes = [ctypes.c_double]
		fortran.krome_photobin_scale_array.restype = None
		fortran.krome_photobin_scale_array.argtypes = [array_1d_double]
		fortran.krome_photobin_restore.restype = None
		fortran.krome_photobin_restore.argtypes = None
		fortran.krome_photobin_restore.restype = None
		fortran.krome_photobin_restore.argtypes = None
		fortran.krome_photobin_store.restype = None
		fortran.krome_photobin_store.argtypes = None
		## extern void krome_load_photoBin_file(char *fname);
		# Here is another example of a Fortran subroutine which accepts optional arguments:
		fortran.krome_set_photobin_hmlog.restype = None
		fortran.krome_set_photobin_hmlog.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_bblin.restype = None
		fortran.krome_set_photobin_bblin.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_bblog.restype = None
		fortran.krome_set_photobin_bblog.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_bblog_auto.restype = None
		fortran.krome_set_photobin_bblog_auto.argtypes = [ctypes.c_double]
		fortran.krome_set_photobin_drainelin.restype = None
		fortran.krome_set_photobin_drainelin.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_drainelog.restype = None
		fortran.krome_set_photobin_drainelog.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_j21lin.restype = None
		fortran.krome_set_photobin_j21lin.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_set_photobin_j21log.restype = None
		fortran.krome_set_photobin_j21log.argtypes = [ctypes.c_double, ctypes.c_double]
		fortran.krome_get_opacity.restype = array_1d_double
		fortran.krome_get_opacity.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_get_opacity_size.restype = array_1d_double
		fortran.krome_get_opacity_size.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double]
		fortran.krome_dump_jflux.restype = None
		fortran.krome_dump_jflux.argtypes = [ctypes.c_int]
		fortran.krome_get_coef.restype = array_1d_double
		fortran.krome_get_coef.argtypes = [ctypes.c_double]
		fortran.krome_get_mu_x.restype = ctypes.c_double
		fortran.krome_get_mu_x.argtypes = [array_1d_double]
		fortran.krome_get_gamma_x.restype = ctypes.c_double
		fortran.krome_get_gamma_x.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_consistent_x.restype = None
		fortran.krome_consistent_x.argtypes = [array_1d_double]
		fortran.krome_n2x.restype = array_1d_double
		fortran.krome_n2x.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_x2n.restype = array_1d_double
		fortran.krome_x2n.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_thermo.restype = None
		fortran.krome_thermo.argtypes = [array_1d_double, array_1d_double, ctypes.c_double]
		fortran.krome_get_heating.restype = ctypes.c_double
		fortran.krome_get_heating.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_get_heating_array.restype = None
		fortran.krome_get_heating_array.argtypes = [array_1d_double, ctypes.c_double, array_1d_double]
		fortran.krome_get_cooling.restype = ctypes.c_double
		fortran.krome_get_cooling.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_get_cooling_array.restype = None
		fortran.krome_get_cooling_array.argtypes = [array_1d_double, ctypes.c_double, array_1d_double]
		fortran.krome_plot_cooling.restype = None
		fortran.krome_plot_cooling.argtypes = [array_1d_double]
		# Here is another example of a Fortran subroutine which accepts optional arguments
		# (and which are subsequently forced to be mandatory...for now).
		fortran.krome_dump_cooling.restype = None
		fortran.krome_dump_cooling.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_int]
		fortran.krome_conservelin_x.restype = None
		fortran.krome_conservelin_x.argtypes = [array_1d_double, array_1d_double]
		fortran.krome_conservelingetref_x.restype = ctypes.c_double
		fortran.krome_conservelingetref_x.argtypes = [array_1d_double]
		fortran.krome_conserve.restype = array_1d_double
		fortran.krome_conserve.argtypes = [array_1d_double, array_1d_double]
		fortran.krome_get_gamma.restype = ctypes.c_double
		fortran.krome_get_gamma.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_get_zatoms.restype = ctypes.c_int
		fortran.krome_get_zatoms.argtypes = None
		fortran.krome_get_mu.restype = ctypes.c_double
		fortran.krome_get_mu.argtypes = [array_1d_double]
		## extern char **krome_get_rnames();
		fortran.krome_get_mass.restype = array_1d_double
		fortran.krome_get_mass.argtypes = None
		fortran.krome_get_imass.restype = array_1d_double
		fortran.krome_get_imass.argtypes = None
		fortran.krome_get_hnuclei.restype = ctypes.c_double
		fortran.krome_get_hnuclei.argtypes = [array_1d_double]
		fortran.krome_get_charges.restype = array_1d_double
		fortran.krome_get_charges.argtypes = None
		## extern char **krome_get_names();
		##extern int krome_get_index(char *name);
		fortran.krome_get_rho.restype = ctypes.c_double
		fortran.krome_get_rho.argtypes = [array_1d_double]
		fortran.krome_scale_z.restype = None
		fortran.krome_scale_z.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_set_z.restype = None
		fortran.krome_set_z.argtypes = [ctypes.c_double]
		fortran.krome_set_clump.restype = None
		fortran.krome_set_clump.argtypes = [ctypes.c_double]
		fortran.krome_get_electrons.restype = ctypes.c_double
		fortran.krome_get_electrons.argtypes = [array_1d_double]
		fortran.krome_print_best_flux.restype = None
		fortran.krome_print_best_flux.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_int]
		fortran.krome_print_best_flux_frac.restype = None
		fortran.krome_print_best_flux_frac.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double]
		fortran.krome_print_best_flux_spec.restype = None
		fortran.krome_print_best_flux_spec.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
		fortran.krome_get_flux.restype = array_1d_double
		fortran.krome_get_flux.argtypes = [array_1d_double, ctypes.c_double]
		fortran.krome_explore_flux.restype = None
		fortran.krome_explore_flux.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_int, ctypes.c_double]
		fortran.krome_get_qeff.restype = array_1d_double
		fortran.krome_get_qeff.argtypes = None
		fortran.krome_dump_flux.restype = None
		fortran.krome_dump_flux.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_int]
		fortran.krome_dump_rates.restype = None
		fortran.krome_dump_rates.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int]
		fortran.krome_get_info.restype = None
		fortran.krome_get_info.argtypes = [array_1d_double, ctypes.c_double]
