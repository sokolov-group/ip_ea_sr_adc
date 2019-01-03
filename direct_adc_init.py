import sys
import numpy as np
from functools import reduce
import pyscf.ao2mo
import direct_adc_spin_integrated.direct_adc_compute as direct_adc_compute


class DirectADC:
    def __init__(self, mf):

        print ("Initializing Direct ADC...\n")

        # General info
        self.mo_a = mf.mo_coeff.copy()
        self.mo_b = mf.mo_coeff.copy()
        self.nmo = self.mo_a.shape[1]
        self.nelec = mf.mol.nelectron
        self.nelec_a = mf.mol.nelectron // 2
        self.nelec_b = mf.mol.nelectron // 2
        self.nocc_a = self.nelec_a
        self.nocc_b = self.nelec_b
        self.nvir_a = self.nmo - self.nelec_a
        self.nvir_b = self.nmo - self.nelec_b
        self.enuc = mf.mol.energy_nuc()
        self.mo_energy_a = mf.mo_energy.copy()
        self.mo_energy_b = mf.mo_energy.copy()
        self.e_scf = mf.e_tot
 
        # Direct ADC specific variables
        self.nstates = 10
        self.step = 0.01
        self.freq_range = (-1.0, 0.0)
        self.broadening = 0.01
        self.tol = 1e-6
        self.maxiter = 200
        self.method = "adc(3)"

        # Integral transformation
        h1e_ao = mf.get_hcore()
        self.h1e_a = reduce(np.dot, (self.mo_a.T, h1e_ao, self.mo_a))
        self.h1e_b = reduce(np.dot, (self.mo_b.T, h1e_ao, self.mo_b))
        


        self.v2e = lambda:None


#        print(v2e_vvvv_a.flags)
#        print(v2e_vvvv_ab.flags)
#        print(v2e_vvvv_b.flags)
       

        occ_a = self.mo_a[:,:self.nocc_a].copy()
        occ_b = self.mo_b[:,:self.nocc_b].copy()
        vir_a = self.mo_a[:,self.nocc_a:].copy()
        vir_b = self.mo_a[:,self.nocc_b:].copy()


        occ = occ_a, occ_b
        vir = vir_a, vir_b

        
        self.v2e.oovv = transform_antisymmetrize_integrals(mf, (occ,occ,vir,vir))
        self.v2e.vvvv = transform_antisymmetrize_integrals(mf, (vir,vir,vir,vir))
        self.v2e.oooo = transform_antisymmetrize_integrals(mf, (occ,occ,occ,occ))
        self.v2e.voov = transform_antisymmetrize_integrals(mf, (vir,occ,occ,vir))
        self.v2e.ooov = transform_antisymmetrize_integrals(mf, (occ,occ,occ,vir))
        self.v2e.vovv = transform_antisymmetrize_integrals(mf, (vir,occ,vir,vir))
        self.v2e.vvoo = transform_antisymmetrize_integrals(mf, (vir,vir,occ,occ))
        self.v2e.vvvo = transform_antisymmetrize_integrals(mf, (vir,vir,vir,occ))
        self.v2e.ovoo = transform_antisymmetrize_integrals(mf, (occ,vir,occ,occ))
        self.v2e.ovov = transform_antisymmetrize_integrals(mf, (occ,vir,occ,vir))
        self.v2e.vooo = transform_antisymmetrize_integrals(mf, (vir,occ,occ,occ))
        self.v2e.oovo = transform_antisymmetrize_integrals(mf, (occ,occ,vir,occ))
        self.v2e.vovo = transform_antisymmetrize_integrals(mf, (vir,occ,vir,occ))
        self.v2e.vvov = transform_antisymmetrize_integrals(mf, (vir,vir,occ,vir))
        
        #print (np.linalg.norm(self.v2e.oovv[0]))
        #exit()
    

    def kernel(self):

        direct_adc_compute.kernel(self)


def transform_antisymmetrize_integrals(mf,mo):

    mo_1, mo_2, mo_3, mo_4 = mo

    mo_1_a, mo_1_b = mo_1
    mo_2_a, mo_2_b = mo_2
    mo_3_a, mo_3_b = mo_3
    mo_4_a, mo_4_b = mo_4

    v2e_a = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_3_a, mo_2_a, mo_4_a), compact=False)
    v2e_a = v2e_a.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_a.shape[1], mo_4_a.shape[1])
    v2e_a = v2e_a.transpose(0,2,1,3).copy()

    if (mo_1_a is mo_2_a):
        v2e_a -= v2e_a.transpose(1,0,2,3)
    elif (mo_3_a is mo_4_a):
        v2e_a -= v2e_a.transpose(0,1,3,2)
    else:
        v2e_temp = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_4_a, mo_2_a, mo_3_a), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_a.shape[1], mo_4_a.shape[1], mo_2_a.shape[1], mo_3_a.shape[1])
        v2e_a -= v2e_temp.transpose(0,2,3,1).copy()


    v2e_b = pyscf.ao2mo.general(mf._eri, (mo_1_b, mo_3_b, mo_2_b, mo_4_b), compact=False)
    v2e_b = v2e_b.reshape(mo_1_b.shape[1], mo_3_b.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_b = v2e_b.transpose(0,2,1,3).copy()

    if (mo_1_b is mo_2_b):
        v2e_b -= v2e_b.transpose(1,0,2,3)
    elif (mo_3_b is mo_4_b):
        v2e_b -= v2e_b.transpose(0,1,3,2)
    else:
        v2e_temp = pyscf.ao2mo.general(mf._eri, (mo_1_b, mo_4_b, mo_2_b, mo_3_b), compact=False)
        v2e_temp = v2e_temp.reshape(mo_1_b.shape[1], mo_4_b.shape[1], mo_2_b.shape[1], mo_3_b.shape[1])
        v2e_b -= v2e_temp.transpose(0,2,3,1).copy()

    v2e_ab = pyscf.ao2mo.general(mf._eri, (mo_1_a, mo_3_a, mo_2_b, mo_4_b), compact=False)
    v2e_ab = v2e_ab.reshape(mo_1_a.shape[1], mo_3_a.shape[1], mo_2_b.shape[1], mo_4_b.shape[1])
    v2e_ab = v2e_ab.transpose(0,2,1,3).copy()

    return (v2e_a, v2e_ab, v2e_b)


