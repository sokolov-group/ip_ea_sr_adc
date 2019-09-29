#####################################
### interface---IP/EA-ADC methods ###
#####################################
import sys
import numpy as np
from functools import reduce
import pyscf.ao2mo
import pyscf.lib
import direct_adc_spin_integrated.direct_adc_compute as direct_adc_compute
import direct_adc_spin_integrated.disk_helper as disk_helper

class DirectADC:
    def __init__(self, mf):

        print ("Initializing Direct ADC...\n")

	## Storing MOs, MO energies and number of electrons for closed-shell systems ##
        if "RHF" in str(type(mf)):
            print ("RHF reference detected")

            mo = mf.mo_coeff.copy()
            self.nmo_a = mo.shape[1]
            self.nmo_b = mo.shape[1]

            for p in range(self.nmo_a):
                max_coeff_ind = np.argmax(np.absolute(mo[:,p]))
                if (mo[max_coeff_ind,p] < 0.0):
                    mo[:,p] *= -1.0

            self.mo_a = mo.copy()
            self.mo_b = mo.copy()
            self.nelec = mf.mol.nelectron
            self.nelec_a = mf.mol.nelectron // 2
            self.nelec_b = mf.mol.nelectron // 2
            self.nocc_a = self.nelec_a
            self.nocc_b = self.nelec_b
            self.nvir_a = self.nmo_a - self.nelec_a
            self.nvir_b = self.nmo_b - self.nelec_b
            self.enuc = mf.mol.energy_nuc()
            self.mo_energy_a = mf.mo_energy.copy()
            self.mo_energy_b = mf.mo_energy.copy()
            self.e_scf = mf.e_tot

	## Storing MOs, MO energies and number of electrons for open-shell systems ##
        elif "UHF" in str(type(mf)):
            print ("UHF reference detected")

            mo_a = mf.mo_coeff[0].copy()
            mo_b = mf.mo_coeff[1].copy()
            self.nmo_a = mo_a.shape[1]
            self.nmo_b = mo_b.shape[1]

            for p in range(self.nmo_a):
                max_coeff_ind = np.argmax(np.absolute(mo_a[:,p]))
                if (mo_a[max_coeff_ind,p] < 0.0):
                    mo_a[:,p] *= -1.0

            for p in range(self.nmo_b):
                max_coeff_ind = np.argmax(np.absolute(mo_b[:,p]))
                if (mo_b[max_coeff_ind,p] < 0.0):
                    mo_b[:,p] *= -1.0

            self.mo_a = mo_a.copy()
            self.mo_b = mo_b.copy()
            self.nelec = mf.mol.nelectron
            self.nelec_a = mf.nelec[0]
            self.nelec_b = mf.nelec[1]
            self.nocc_a = self.nelec_a
            self.nocc_b = self.nelec_b
            self.nvir_a = self.nmo_a - self.nelec_a
            self.nvir_b = self.nmo_b - self.nelec_b
            self.enuc = mf.mol.energy_nuc()
            self.mo_energy_a = mf.mo_energy[0].copy()
            self.mo_energy_b = mf.mo_energy[1].copy()
            self.e_scf = mf.e_tot

        else:
            raise Exception("ADC code is not implemented for this reference")

	##########################################################
        ### ADC specific variables for different implementations##
	##########################################################
	#Unless specified otherwise in the input file these are the default parameters

	###Variables common to all implementations###

        # IP or EA flags
        self.EA = True  # by default computes IP as well as EA
        self.IP = True
        self.method = "adc(3)" # Order of ADC used. Can be ADC(2), ADC(3) or ADC(2)-E
        self.algorithm = "conventional" # Implementation used. Can be conventional, GF (for Green's Function) or CVS
        self.disk = False    # Whether to use disk

	### Conventional ADC ###
        self.davidson = pyscf.lib.linalg_helper.davidson #Use of Davidson iterative algorithm for solving eigenvalue equation
        self.nstates = 10    # Number of states requested
        self.verbose = 6     # Verbose level
        self.max_cycle = 150 # Maximum number of iterations
        self.max_space = 12  # Space size to hold trial vectors

	### Green's Function ADC ###
        self.step = 0.01 # Step size
        self.freq_range = (-1.0, 0.0) # Frequency range
        self.broadening = 0.01        # Broadening parameter
        self.tol = 1e-6               #Threshold for linear equations convergence
        self.maxiter = 200            #Maximum number of iterations
        self.freq_outer_loop = True   #True if frequency forms the outer loop. If False, orbitals form the outer loop

	### CVS-ADC for core-ionization ###
        self.n_core = 5 # number of core spatial orbitals

	#################################
	########## Integrals ############
	#################################
        self.v2e = lambda:None
        self.h1e_ao = mf.get_hcore()
        if mf._eri is None:
            self.v2e_ao = mf.mol
        else:
            self.v2e_ao = mf._eri

    def kernel(self):

        self.transform_integrals()
        if self.algorithm == "GF":
            direct_adc_compute.kernel(self)
        elif self.algorithm == "conventional" or self.algorithm == "cvs":
            direct_adc_compute.conventional(self)
        else:
            raise Exception("Algorithm is not recognized")

        if self.disk:
            disk_helper.remove_dataset(self.v2e.vvvv[0])
            disk_helper.remove_dataset(self.v2e.vvvv[1])
            disk_helper.remove_dataset(self.v2e.vvvv[2])

    ### Integral transformation ###
    def transform_integrals(self):
        h1e_ao = self.h1e_ao
        self.h1e_a = reduce(np.dot, (self.mo_a.T, h1e_ao, self.mo_a))
        self.h1e_b = reduce(np.dot, (self.mo_b.T, h1e_ao, self.mo_b))

        occ_a = self.mo_a[:,:self.nocc_a].copy()
        occ_b = self.mo_b[:,:self.nocc_b].copy()
        vir_a = self.mo_a[:,self.nocc_a:].copy()
        vir_b = self.mo_b[:,self.nocc_b:].copy()

        occ = occ_a, occ_b
        vir = vir_a, vir_b

        self.v2e.oovv = transform_antisymmetrize_integrals(self.v2e_ao, (occ,occ,vir,vir))
        self.v2e.vvvv = transform_antisymmetrize_integrals(self.v2e_ao, (vir,vir,vir,vir))
        self.v2e.oooo = transform_antisymmetrize_integrals(self.v2e_ao, (occ,occ,occ,occ))
        self.v2e.voov = transform_antisymmetrize_integrals(self.v2e_ao, (vir,occ,occ,vir))
        self.v2e.ooov = transform_antisymmetrize_integrals(self.v2e_ao, (occ,occ,occ,vir))
        self.v2e.vovv = transform_antisymmetrize_integrals(self.v2e_ao, (vir,occ,vir,vir), self.disk)
        self.v2e.vvoo = transform_antisymmetrize_integrals(self.v2e_ao, (vir,vir,occ,occ))
        self.v2e.vvvo = transform_antisymmetrize_integrals(self.v2e_ao, (vir,vir,vir,occ), self.disk)
        self.v2e.ovoo = transform_antisymmetrize_integrals(self.v2e_ao, (occ,vir,occ,occ))
        self.v2e.ovov = transform_antisymmetrize_integrals(self.v2e_ao, (occ,vir,occ,vir))
        self.v2e.vooo = transform_antisymmetrize_integrals(self.v2e_ao, (vir,occ,occ,occ))
        self.v2e.oovo = transform_antisymmetrize_integrals(self.v2e_ao, (occ,occ,vir,occ))
        self.v2e.vovo = transform_antisymmetrize_integrals(self.v2e_ao, (vir,occ,vir,occ))
        self.v2e.vvov = transform_antisymmetrize_integrals(self.v2e_ao, (vir,vir,occ,vir), self.disk)
        self.v2e.ovvo = transform_antisymmetrize_integrals(self.v2e_ao, (occ,vir,vir,occ))
        self.v2e.ovvv = transform_antisymmetrize_integrals(self.v2e_ao, (occ,vir,vir,vir), self.disk)

def transform_antisymmetrize_integrals(v2e_ao, mo, disk = False):

    mo_1, mo_2, mo_3, mo_4 = mo

    mo_1_a, mo_1_b = mo_1
    mo_2_a, mo_2_b = mo_2
    mo_3_a, mo_3_b = mo_3
    mo_4_a, mo_4_b = mo_4

    v2e_a = None
#    if mf._eri is None:
#        v2e_a = pyscf.ao2mo.general(mf.mol, (mo_1_a, mo_3_a, mo_2_a, mo_4_a), compact=False)
#    else :
#        v2e_a = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_3_a, mo_2_a, mo_4_a), compact=False)
    v2e_a = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_3_a, mo_2_a, mo_4_a), compact=False)
    v2e_a = v2e_a.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_a.shape[1], mo_4_a.shape[1])
    v2e_a = v2e_a.transpose(0,2,1,3).copy()

    if (mo_1_a is mo_2_a):
        v2e_a -= v2e_a.transpose(1,0,2,3).copy()
    elif (mo_3_a is mo_4_a):
        v2e_a -= v2e_a.transpose(0,1,3,2).copy()
    else:
        v2e_temp = None
#        if mf._eri is None:
#            v2e_temp = pyscf.ao2mo.general(mf.mol, (mo_1_a, mo_4_a, mo_2_a, mo_3_a), compact=False)
#        else :
#            v2e_temp = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_4_a, mo_2_a, mo_3_a), compact=False)
        v2e_temp = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_4_a, mo_2_a, mo_3_a), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_a.shape[1], mo_4_a.shape[1], mo_2_a.shape[1], mo_3_a.shape[1])
        v2e_a -= v2e_temp.transpose(0,2,3,1).copy()
        del v2e_temp

    v2e_a = disk_helper.dataset(v2e_a) if disk else v2e_a

    v2e_b = None
#    if mf._eri is None:
#        v2e_b = pyscf.ao2mo.general(mf.mol, (mo_1_b, mo_3_b, mo_2_b, mo_4_b), compact=False)
#    else:
#        v2e_b = pyscf.ao2mo.general(mf._eri, (mo_1_b, mo_3_b, mo_2_b, mo_4_b), compact=False)
    v2e_b = pyscf.ao2mo.general(v2e_ao, (mo_1_b, mo_3_b, mo_2_b, mo_4_b), compact=False)
    v2e_b = v2e_b.reshape(mo_1_b.shape[1], mo_3_b.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_b = v2e_b.transpose(0,2,1,3).copy()

    if (mo_1_b is mo_2_b):
        v2e_b -= v2e_b.transpose(1,0,2,3).copy()
    elif (mo_3_b is mo_4_b):
        v2e_b -= v2e_b.transpose(0,1,3,2).copy()
    else:
        v2e_temp = None
#        if mf._eri is None :
#            v2e_temp = pyscf.ao2mo.general(mf.mol, (mo_1_b, mo_4_b, mo_2_b, mo_3_b), compact=False)
#        else : 
#            v2e_temp = pyscf.ao2mo.general(mf._eri, (mo_1_b, mo_4_b, mo_2_b, mo_3_b), compact=False)
        v2e_temp = pyscf.ao2mo.general(v2e_ao, (mo_1_b, mo_4_b, mo_2_b, mo_3_b), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_b.shape[1], mo_4_b.shape[1], mo_2_b.shape[1], mo_3_b.shape[1])
        v2e_b -= v2e_temp.transpose(0,2,3,1).copy()
        del v2e_temp

    v2e_b = disk_helper.dataset(v2e_b) if disk else v2e_b

    v2e_ab = None
#    if mf._eri is None :
#        v2e_ab = pyscf.ao2mo.general(mf.mol, (mo_1_a, mo_3_a, mo_2_b, mo_4_b), compact=False)
#    else :
#        v2e_ab = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_3_a, mo_2_b, mo_4_b), compact=False)
    v2e_ab = pyscf.ao2mo.general(v2e_ao, (mo_1_a, mo_3_a, mo_2_b, mo_4_b), compact=False)
    v2e_ab = v2e_ab.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_ab = v2e_ab.transpose(0,2,1,3).copy()

    v2e_ab = disk_helper.dataset(v2e_ab) if disk else v2e_ab

    return (v2e_a, v2e_ab, v2e_b)
