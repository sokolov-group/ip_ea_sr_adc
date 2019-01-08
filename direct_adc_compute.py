import sys
import numpy as np
import time
from functools import reduce

def kernel(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

    print ("\nStarting spin-orbital direct ADC code..\n")
    print ("Number of electrons:               ", direct_adc.nelec)
    print ("Number of alpha electrons:         ", direct_adc.nelec_a)
    print ("Number of beta electrons:          ", direct_adc.nelec_b)
    print ("Number of basis functions:         ", direct_adc.nmo)
    print ("Number of alpha occupied orbitals: ", direct_adc.nocc_a)
    print ("Number of beta occupied orbitals:  ", direct_adc.nocc_b)
    print ("Number of alpha virtual orbitals:  ", direct_adc.nvir_a)
    print ("Number of beta virtual orbitals:   ", direct_adc.nvir_b)
    print ("Nuclear repulsion energy:          ", direct_adc.enuc,"\n")
    

    print ("Number sin_aof states:            ", direct_adc.nstates)
    print ("Frequency step:              ", direct_adc.step)
    print ("Frequency range:             ", direct_adc.freq_range)
    print ("Broadening:                  ", direct_adc.broadening)
    print ("Tolerance:                   ", direct_adc.tol)
    print ("Maximum number of iterations:", direct_adc.maxiter, "\n")
    
    print ("SCF orbital energies(alpha):\n", direct_adc.mo_energy_a, "\n")
    print ("SCF orbital energies(beta):\n", direct_adc.mo_energy_b, "\n")

    t_start = time.time()

    # Compute amplitudes
    t_amp = compute_amplitudes(direct_adc)
    
    # Compute MP2 energy
    e_mp2 = compute_mp2_energy(direct_adc, t_amp)

     
    #print ("MP2 correlation energy:  ", e_mp2)
    #print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")

    # Compute the sigma vector
    apply_H = define_H(direct_adc,t_amp)

    # Compute Green's functions directly
    dos = calc_density_of_states(direct_adc, apply_H, t_amp)

    #print ("Computation successfully finished")
    #print ("Total time:", (time.time() - t_start, "sec"))

    # Save to a file or plot depeneding on user input
    # Plot     


    
    
def compute_amplitudes(direct_adc):

    t2_1, t2_2, t1_2, t1_3 = (None,) * 4

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b  = direct_adc.v2e.oovv
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b  = direct_adc.v2e.vvvv
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b  = direct_adc.v2e.oooo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b  = direct_adc.v2e.voov
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b  = direct_adc.v2e.ooov
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b  = direct_adc.v2e.vovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b  = direct_adc.v2e.vvoo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b  = direct_adc.v2e.oovo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b  = direct_adc.v2e.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b  = direct_adc.v2e.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b  = direct_adc.v2e.vvvo
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b  = direct_adc.v2e.vvov
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b  = direct_adc.v2e.vooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b  = direct_adc.v2e.ovoo
    
    e_a = direct_adc.mo_energy_a
    e_b = direct_adc.mo_energy_b

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]
    


    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]
    
    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
    
    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))

        
    D1_a = e_a[:nocc_a][:None].reshape(-1,1) - e_a[nocc_a:].reshape(-1)
    D1_b = e_b[:nocc_b][:None].reshape(-1,1) - e_b[nocc_b:].reshape(-1)
    D1_a = D1_a.reshape((nocc_a,nvir_a))
    D1_b = D1_b.reshape((nocc_b,nvir_b))
    
    
    t2_1_a = v2e_oovv_a/D2_a
    t2_1_b = v2e_oovv_b/D2_b
    t2_1_ab = v2e_oovv_ab/D2_ab


    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)    

    
    t1_2_a = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_a,t2_1_a)
    t1_2_a -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_a,t2_1_a)
    
    t1_2_a += np.einsum('akcd,ikcd->ia',v2e_vovv_ab,t2_1_ab)
    t1_2_a -= np.einsum('klic,klac->ia',v2e_ooov_ab,t2_1_ab)
    
    
    t1_2_b = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_b,t2_1_b)
    t1_2_b -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_b,t2_1_b)
    
    t1_2_b += np.einsum('akcd,ikcd->ia',v2e_vovv_ab,t2_1_ab)
    t1_2_b -= np.einsum('klic,klac->ia',v2e_ooov_ab,t2_1_ab)
    
    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b


    t1_2 = (t1_2_a , t1_2_b)


    if (direct_adc.method == "adc(2)-e" or direct_adc.method == "adc(3)"):

        #print("Calculating additional amplitudes for adc(2)-e and adc(3)")


        t2_2_a = 0.5*np.einsum('abcd,ijcd->ijab',v2e_vvvv_a,t2_1_a)
        t2_2_a += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_a,t2_1_a)
                 

        temp = np.einsum('bkjc,kica->ijab',v2e_voov_a,t2_1_a) 
        temp_1 = np.einsum('bkjc,kica->ijab',v2e_voov_ab,t2_1_ab) 
        
        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
        
    
        t2_2_b = 0.5*np.einsum('abcd,ijcd->ijab',v2e_vvvv_b,t2_1_b)
        t2_2_b += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_b,t2_1_b)
        

        temp = np.einsum('bkjc,kica->ijab',v2e_voov_b,t2_1_b) 
        temp_1 = np.einsum('bkjc,kica->ijab',v2e_voov_ab,t2_1_ab) 
        
        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
   

        t2_2_ab = np.einsum('abcd,ijcd->ijab',v2e_vvvv_ab,t2_1_ab)
        t2_2_ab += np.einsum('klij,klab->ijab',v2e_oooo_ab,t2_1_ab)
        

        t2_2_ab += np.einsum('bkjc,kica->ijab',v2e_voov_ab,t2_1_a) 
        t2_2_ab += np.einsum('bkjc,kica->ijab',v2e_voov_b,t2_1_ab) 
        
        t2_2_ab -= np.einsum('kbic,kjac->ijab',v2e_ovov_ab,t2_1_ab) 
        
        t2_2_ab -= np.einsum('akcj,ikcb->ijab',v2e_vovo_ab,t2_1_ab) 

        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_ab,t2_1_b)
        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_a,t2_1_ab)

        
        t2_2_a = t2_2_a/D2_a
        t2_2_b = t2_2_b/D2_b
        t2_2_ab = t2_2_ab/D2_ab
        

        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)    

    if (direct_adc.method == "adc(3)"):
         
        #print("Calculating additional amplitudes for adc(3)")
        

        t1_3_a = np.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a)
        t1_3_a += np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_b)
        
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a,t1_2_a)
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_b)
        
        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a)
        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_b)
        
        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a,t1_2_a)
        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_b)
        
        t1_3_a += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_a)
        t1_3_a += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_ab)
        
        t1_3_a += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_a)
        t1_3_a += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_ab)

        t1_3_a -= 0.5*np.einsum('lmad,lmid->ia',t2_2_a,v2e_ooov_a)
        t1_3_a -= np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab)
        
        t1_3_a += 0.5*np.einsum('ilde,alde->ia',t2_2_a,v2e_vovv_a)
        t1_3_a += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab)
        
        #t1_3_a = -np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a)
        #t1_3_a += np.einsum('ildf,aefm,lmde->ia',t2_1_ab,v2e_vvvo_a,t2_1_ab)
        #t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab)
        #t1_3_a += np.einsum('ildf,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_b)
        #t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab)

        #print (np.linalg.norm(t1_3_a))
        #sys.exit()      

        #t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a)
        #t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_b)
        #t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab)
        #t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab)
        


        #t1_3_a += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        #t1_3_a += np.einsum('inde,anlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        

        #t1_3_a = +0.5*np.einsum('inad,enlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        #t1_3_a -= np.einsum('inad,enlm,lmde->ia',t2_1_a,v2e_vooo_ab,t2_1_ab)
        #t1_3_a -= np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        #t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_b)
        
        #print (np.linalg.norm(t1_3_a))
        #sys.exit()      

        #t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        #t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_a,t2_1_ab)
        #t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_ab,t2_1_b)
        #t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)


        #t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_a,v2e_vvov_a,t2_1_a)
        #t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_a,t2_1_ab)
        #t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_ab,t2_1_b)
        #t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        
        #t1_3_a = -np.einsum('lnde,emin,lmad->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        #t1_3_a += np.einsum('lnde,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_a)
        #t1_3_a += np.einsum('lnde,emin,lmad->ia',t2_1_ab,v2e_vooo_a,t2_1_ab)
        #t1_3_a += np.einsum('lnde,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        #t1_3_a -= np.einsum('lnde,emin,lmad->ia',t2_1_b,v2e_vooo_ab,t2_1_ab)

        #t1_3_a -= np.einsum('lnde,emin,lmad->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        #t1_3_a -= np.einsum('lnde,mein,lmad->ia',t2_1_ab,v2e_ovoo_ab,t2_1_a)
        #t1_3_a += np.einsum('lnde,emin,lmad->ia',t2_1_ab,v2e_vooo_a,t2_1_ab)
        #t1_3_a += np.einsum('lnde,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        #t1_3_a += np.einsum('lnde,mein,lmad->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab)
        

        #print (np.linalg.norm(t1_3_a))
        #sys.exit()      
        

        #t1_3_a -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_a,v2e_vvov_a,t2_1_a)
        #t1_3_a -= np.einsum('lmef,efid,lmad->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        

        t1_3_b = np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_2_b,t1_2_b)
        t1_3_b += np.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_2_ab,t1_2_a)
        
        t1_3_b -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_2_b,t1_2_b)
        t1_3_b -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_2_ab,t1_2_a)
        
        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_2_b,t1_2_b)
        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_2_ab,t1_2_a)
        
        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_2_b,t1_2_b)
        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_2_ab,t1_2_a)
        
        t1_3_b += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_b)
        t1_3_b += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_ab)
        
        t1_3_b += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_b)
        t1_3_b += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_ab)

        t1_3_b -= 0.5*np.einsum('lmad,lmid->ia',t2_2_b,v2e_ooov_b)
        t1_3_b -= np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab)
        
        t1_3_b += 0.5*np.einsum('ilde,alde->ia',t2_2_b,v2e_vovv_b)
        t1_3_b += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab)
        
        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b


    
    
        t1_3 = (t1_3_a , t1_3_b)    


    t_amp = (t2_1, t2_2, t1_2, t1_3)

    return t_amp

def compute_mp2_energy(direct_adc, t_amp):



    v2e_oovv_a, v2e_oovv_ab, v2e_oovv_b = direct_adc.v2e.oovv
    
    t2_1_a, t2_1_ab, t2_1_b  = t_amp[0]
    
    e_mp2 = 0.25 * np.einsum('ijab,ijab', t2_1_a, v2e_oovv_a)
    e_mp2 += np.einsum('ijab,ijab', t2_1_ab, v2e_oovv_ab)
    e_mp2 += 0.25 * np.einsum('ijab,ijab', t2_1_b, v2e_oovv_b)
  
    #print(e_mp2)
    return e_mp2


def calc_density_of_states(direct_adc,apply_H,t_amp):

	nmo_a = direct_adc.nmo
	nmo_b = direct_adc.nmo
	nmo = nmo_a+nmo_b
	freq_range = direct_adc.freq_range	
	broadening = direct_adc.broadening
	step = direct_adc.step

	freq_range = np.arange(freq_range[0],freq_range[1],step)


	k = np.zeros((nmo,nmo))
	gf = np.array(k,dtype = complex)
	gf.imag = k

	gf_trace= []
	gf_im_trace = [] 

	for freq in freq_range:
       
		omega = freq
		iomega = freq + broadening*1j


		for orb in range(nmo):

			T = calculate_T(direct_adc, t_amp, orb)
			#print (np.linalg.norm(T))
			gf[orb,orb] = calculate_GF(direct_adc,apply_H,omega,orb,T)

		gf_trace = -(1/(np.pi))*np.trace(gf.imag)
		gf_im_trace.append(gf_trace)
        

	return gf_im_trace         


def calculate_T(direct_adc, t_amp, orb):

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2


    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2
    n_doubles_abb = nocc_a* (nocc_b) * nvir_b 
    n_doubles_baa = nocc_b* (nocc_a) * nvir_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2
    
    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_abb + n_doubles_baa + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)
    
    v2e_oovv_a , v2e_oovv_ab, v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vvvo_a , v2e_vvvo_ab, v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_ovoo_a , v2e_ovoo_ab, v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_voov_a , v2e_voov_ab, v2e_voov_b = direct_adc.v2e.voov
    v2e_ovov_a , v2e_ovov_ab, v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vovv_a , v2e_vovv_ab, v2e_vovv_b = direct_adc.v2e.vovv
    v2e_ooov_a , v2e_ooov_ab, v2e_ooov_b = direct_adc.v2e.ooov
    
    sin_a = n_singles_a
    sin_b = sin_a + n_singles_b
    d_aaa = sin_b + n_doubles_aaa
    d_abb = d_aaa + n_doubles_abb
    d_baa = d_abb + n_doubles_baa
    d_bbb = d_baa + n_doubles_bbb
    

    T = np.zeros((dim))
    T1_a = T[:sin_a]
    T1_b = T[sin_a:sin_b]
    T2_aaa = T[sin_b:d_aaa] 
    T2_abb = T[d_aaa:d_abb] 
    T2_baa = T[d_abb:d_baa] 
    T2_bbb = T[d_baa:d_bbb] 

    occ_a = nocc_a
    occ_b = occ_a + nocc_b
    vir_a = occ_b + nvir_a
    vir_b = vir_a + nvir_b 

    ij_ind_a = np.tril_indices(nocc_a,k=-1)
    ij_ind_b = np.tril_indices(nocc_b,k=-1)
    
    #ADC(2) 1h part
   
   # if orb < occ_a:
   #     
   #     T1_a = idn_occ_a[orb, :]
   #     T1_a += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
   #     T1_a -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
   #     T1_a -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
   #     
   # elif orb < occ_b :
   #     
   #     T1_b = idn_occ_b[orb, :]
   #     T1_b += 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_1_b, optimize = True)
   #     T1_b -= 0.25*np.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
   #     T1_b -= 0.25*np.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
        
        
    if orb < occ_a :


        #T1_a += t1_2_a[:,(orb-occ_b)]
        T1_a = t1_2_a[:,orb]
     
        #ADC(2) 2h-1p part

        t2_1_t_a = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
        t2_1_t_ab = t2_1_ab[ij_ind_a[0],ij_ind_a[1],:,:].copy()
        
        #T2_aaa = t2_1_t_a[:,:,(orb-occ_b)].T.reshape(-1)
        #T2_baa = t2_1_t_ab[:,:,(orb-occ_b)].T.reshape(-1)
         
        T2_aaa = t2_1_t_a[:,:,orb].T.reshape(-1)
        T2_baa = t2_1_t_ab[:,:,orb].T.reshape(-1)
    
    elif orb < vir_b : 

 
        T1_b += t1_2_b[:,(orb-vir_a)]
     
        #ADC(2) 2h-1p part

        t2_1_t_b = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
        t2_1_t_ab = t2_1_ab[ij_ind_b[0],ij_ind_b[1],:,:].copy()
        
        T2_bbb = t2_1_t_b[:,:,(orb-vir_a)].T.reshape(-1)
        T2_abb = -t2_1_t_ab[:,:,(orb-vir_a)].T.reshape(-1)

#        
#
#     
#    if(method=='adc(2)-e'or method=='adc(3)'):
#      
#    
#    #ADC(3) 2h-1p part 
#        
#        if orb >= nocc_so:
#    
#            t2_2_t = t2_2[ij_ind[0],ij_ind[1],:,:].copy()
#            T2 += t2_2_t[:,:,(orb-nocc_so)].T.reshape(-1)
#    
#
#
#
#    if(method=='adc3'):
#    	
#    
#    #ADC(3) 1h part 
#        
#        if orb < nocc_so:
#            T1 += 0.25*np.einsum('kdc,ikdc->i',t2_1[:,orb,:,:], t2_2, optimize = True) 
#            T1 += 0.25*np.einsum('ikdc,kdc->i',t2_1, t2_2[:,orb,:,:],optimize = True) 
# 
#        else: 
#    
#            T1 += 0.5*np.einsum('ikc,kc->i',t2_1[:,:,(orb-nocc_so),:], t1_2,optimize = True)
#            T1 += t1_3[:,(orb-nocc_so)]
#
    T = np.concatenate((T1_a,T2_aaa))

    return T


def get_Mij(direct_adc,t_amp):


    method = direct_adc.method
   

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t2_2_a, t2_2_ab, t2_2_b = t2_2
    t1_2_a, t1_2_b = t1_2
    #t1_3_a, t1_3_b = t1_3
    

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b


    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)



    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = direct_adc.v2e.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov
    

    
    # i-j block
    # Zeroth-order terms
#    M_ij_a = np.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
#    M_ij_b = np.einsum('ij,j->ij', idn_occ_b ,e_occ_b)
#
#
#
#    # Second-order terms
#
#    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a)
#    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab)
#    M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab)
#    
#    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b)
#    M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab)
#    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_ab, t2_1_ab)
#
#    M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
#    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#
#    M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
#    M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#    M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#
#
#    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
#    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#
#
#    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
#    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#
#    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
#    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
#
#
#    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
#    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
#
#
#    M_ij_a += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_a, v2e_oovv_a)
#    M_ij_a += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)
#
#    M_ij_b += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_b, v2e_oovv_b)
#    M_ij_b += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)
#
#    M_ij_a += 0.5 *  np.einsum('jlde,deil->ij',t2_1_a, v2e_vvoo_a)
#    M_ij_a += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)
#
#    M_ij_b += 0.5 *  np.einsum('jlde,deil->ij',t2_1_b, v2e_vvoo_b)
#    M_ij_b += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)

    # Third-order terms
    

    if (method == "adc(3)"):

        M_ij_a += np.einsum('ld,jlid->ij',t1_2_a, v2e_ooov_a)
        M_ij_a += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_ab)
    

        M_ij_b += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_b)
        M_ij_b += np.einsum('ld,jlid->ij',t1_2_a, v2e_ooov_ab)
        

        M_ij_a += np.einsum('ld,jdil->ij',t1_2_a, v2e_ovoo_a)
        M_ij_a += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_ab)


        M_ij_b += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_b)
        M_ij_b += np.einsum('ld,jdil->ij',t1_2_a, v2e_ovoo_ab)


        M_ij_a += 0.5* np.einsum('ilde,jlde->ij',t2_2_a, v2e_oovv_a)
        M_ij_a += np.einsum('ilde,jlde->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_b += 0.5* np.einsum('ilde,jlde->ij',t2_2_b, v2e_oovv_b)
        M_ij_b += np.einsum('ilde,jlde->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_a += 0.5* np.einsum('jlde,deil->ij',t2_2_a, v2e_vvoo_a)
        M_ij_a += np.einsum('jlde,deil->ij',t2_2_ab, v2e_vvoo_ab)

        M_ij_b += 0.5* np.einsum('jlde,deil->ij',t2_2_b, v2e_vvoo_b)
        M_ij_b += np.einsum('jlde,deil->ij',t2_2_ab, v2e_vvoo_ab)



        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a)
        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab)
        M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab)
        
        M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b)
        M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab)
        M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_ab, t2_2_ab)


        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a)
        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab)
        M_ij_a +=  np.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab)
        
        M_ij_b +=  np.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b)
        M_ij_b +=  np.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab)
        M_ij_b +=  np.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_ab, t2_2_ab)
        

        M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)
    
        M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)

        M_ij_a -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)
    
        M_ij_b -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)


        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)
    
    
        M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)

        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)
    
    
        M_ij_b -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)


        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab)
    
    
        M_ij_b -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab)


        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab)
    
    
        M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b)
        M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)
        M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab)


        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('lmde,jlfd,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,jldf,mfie->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,jlfd,fmie->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)


        M_ij_b -= np.einsum('lmde,jldf,fmie->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,jlfd,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('lmde,jldf,fmie->ij',t2_1_ab, t2_1_b, v2e_voov_ab,optimize = True)
        M_ij_b -= np.einsum('mlde,jldf,mfie->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,jlfd,fmie->ij',t2_1_a, t2_1_ab, v2e_voov_ab ,optimize = True)


        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('lmde,ilfd,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,ildf,mfje->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,ilfd,fmje->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)
        

        M_ij_b -= np.einsum('lmde,ildf,fmje->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,ilfd,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('lmde,ildf,fmje->ij',t2_1_ab, t2_1_b, v2e_voov_ab,optimize = True)
        M_ij_b -= np.einsum('mlde,ildf,mfje->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,ilfd,fmje->ij',t2_1_a, t2_1_ab, v2e_voov_ab ,optimize = True)



        M_ij_a += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a += np.einsum('lmde,jnde,lmin->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)


        M_ij_b += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b += np.einsum('lmde,jnde,lmin->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)


        M_ij_a += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a,v2e_vvvv_a, optimize = True)
        M_ij_a +=np.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b,v2e_vvvv_b, optimize = True)
        M_ij_b +=np.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)


        M_ij_a += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a +=np.einsum('inde,lmde,jnlm->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b +=np.einsum('inde,lmde,jnlm->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_a += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_a, t2_1_a, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab , optimize = True)
        M_ij_a +=0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_ab , optimize = True)


        M_ij_b += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab , optimize = True)
        M_ij_b +=0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_a, t2_1_a, v2e_ovov_ab , optimize = True)


        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_a, t2_1_a, v2e_vovo_a, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lfem->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,flme->ij',t2_1_ab, t2_1_a, v2e_voov_ab, optimize = True)
        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize = True)
        M_ij_a -= np.einsum('iled,jmfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize = True)

        M_ij_b -= np.einsum('ilde,jmdf,flem->ij',t2_1_b, t2_1_b, v2e_vovo_b, optimize = True)
        M_ij_b += np.einsum('ilde,jmdf,lfem->ij',t2_1_b, t2_1_ab, v2e_ovvo_ab, optimize = True)
        M_ij_b += np.einsum('ilde,jmdf,flme->ij',t2_1_ab, t2_1_b, v2e_voov_ab, optimize = True)
        M_ij_b -= np.einsum('ilde,jmdf,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize = True)
        M_ij_b -= np.einsum('iled,jmfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize = True)


        M_ij_a -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_a, t2_1_a, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_a -= 0.5 * np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_ab, optimize = True)

        M_ij_b -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_b -= 0.5 * np.einsum('lnde,lmde,jnim->ij',t2_1_a, t2_1_a, v2e_oooo_ab, optimize = True)
    
    M_ij = (M_ij_a, M_ij_b)
    return M_ij





def define_H(direct_adc,t_amp):

    method = direct_adc.method


    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t2_2_a, t2_2_ab, t2_2_b = t2_2
    t1_2_a, t1_2_b = t1_2
    t1_3_a, t1_3_b = t1_3
    

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

#    sin_a = n_singles_a
#    sin_b = sin_a + n_singles_b
#    d_aaa = sin_b + n_doubles_aaa
#    d_abb = d_aaa + n_doubles_abb
#    d_baa = d_abb + n_doubles_baa
#    d_bbb = d_baa + n_doubles_bbb
    
    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b = direct_adc.v2e.vooo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = direct_adc.v2e.oovo
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = direct_adc.v2e.vovv
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = direct_adc.v2e.vvvo

    v2e_vooo_1_a = v2e_vooo_a.transpose(1,0,2,3).reshape(nocc_a, -1)
    v2e_vooo_1_ab = v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_a, -1)
#   v2e_vooo_1_ab = v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_so, -1)
#    v2e_oovo_1 = v2e_so_oovo[ij_ind[0],ij_ind[1],:,:].transpose(1,0,2)

#    v2e_vooo_1 = v2e_so_vooo[:,:,ij_ind[0],ij_ind[1]].transpose(1,0,2).reshape(nocc_so, -1)
#    v2e_oovo_1 = v2e_so_oovo[ij_ind[0],ij_ind[1],:,:].transpose(1,0,2)
    
#    d_ij = e_occ_so[:,None] + e_occ_so
#    d_a = e_vir_so[:,None]
#    D_n = -d_a + d_ij.reshape(-1)
#    D_n = D_n.reshape((nvir_so,nocc_so,nocc_so))
#    D_aij = D_n.copy()[:,ij_ind[0],ij_ind[1]].reshape(-1)


    M_ij_a, M_ij_b = get_Mij(direct_adc,t_amp)
    print (np.linalg.norm(M_ij_a))
    print (np.linalg.norm(M_ij_b))
    exit() 

    def sigma_(r):
        
        r2_aaa = r
        r2_bab = r
        #r1_a = r[:sin_a]
        #r1_b = r[sin_a:sin_b]
        #r2_aaa = r[sin_b:d_aaa]
        #r2_bab = r[d_aaa:d_bab]
        #r2_aba = r[d_bab:d_aba]
        #r2_bbb = r[d_baa:d_bbb]

        # ADC(2) ij block
        
#        s1_a = np.einsum('ij,j->i',M_ij_a,r1_a) 
#        s1_b = np.einsum('ij,j->i',M_ij_b,r1_b) 
            
        # ADC(2) i - kja block
        #s1 += np.einsum('jp,p->j', v2e_vooo_1, r2, optimize = True)
        s1_a = 0.5*np.einsum('ip,p->i', v2e_vooo_1_a, r2_aaa, optimize = True)
        s1_a += np.einsum('ip,p->i', v2e_vooo_1_ab, r2_bab, optimize = True)


#        s1_a = 0.5*np.einsum('ip,p->i', v2e_vooo_1_a, r2_aaa, optimize = True)
#        s1_a += np.einsum('ip,p->i', v2e_vooo_1_ab, r2_bab, optimize = True)


#        s1_a += np.einsum('iajk,ajk->i', v2e_ovoo_a, r2_bab, optimize = True)
#        
#        s1_a += np.einsum('jp,p->j', v2e_vooo_1, r2, optimize = True)
#
#        # ADC(2) ajk - i block
#        s2 = np.einsum('api,i->ap', v2e_oovo_1, r1, optimize = True).reshape(-1)
#        
#        # ADC(2) ajk - bil block
#
#        s2 += D_aij * r2

        if (method == "adc(2)-e" or method == "adc(3)"):
        	
                #print("Calculating additional terms for adc(2)-e")	
        
                # ajk - bil block
                temp = np.einsum('ab,xywv->axybvw',idn_vir,v2e_so_oooo,optimize = True)
                
                temp_1 = -np.einsum('wx,byav->axybvw',idn_occ, v2e_so_vovo, optimize = True)
                temp_1 -= temp_1.transpose(0,2,1,3,4,5).copy()
                
                temp_2 = np.einsum('vx,byaw->axybvw',idn_occ, v2e_so_vovo, optimize=True)
                temp_2 -= temp_2.transpose(0,2,1,3,4,5).copy()

                temp += temp_1+temp_2

                temp = temp[:,ij_ind[0],ij_ind[1],:,:,:]
                temp = temp[:,:,:,ij_ind[0],ij_ind[1]].reshape(n_doubles,n_doubles)
                

                s2  += np.einsum('pq,q->p',temp, r2, optimize = True)

        if (method == "adc(3)"):
        	
                #print("Calculating additional terms for adc(3)")	
                 

               
                # ADC(3) i - kja block


                temp = 0.5*np.einsum('vwde,bjde->jbvw',t2_1,v2e_so_vovv)
                temp = temp[:,:,ij_ind[0],ij_ind[1]].reshape(nocc_so,-1)
                
                temp_1 = np.einsum('vlbd,jlwd->jbvw',t2_1,v2e_so_ooov)
                temp_2 = temp_1.transpose(0,1,3,2).copy()    

                temp_1 = temp_1[:,:,ij_ind[0],ij_ind[1]].reshape(nocc_so,-1)
                temp_2 = temp_2[:,:,ij_ind[0],ij_ind[1]].reshape(nocc_so,-1)


                s1 += np.einsum('jp,p->j',temp, r2, optimize=True)
                s1 += np.einsum('jp,p->j',temp_1, r2, optimize=True)
                s1 -= np.einsum('jp,p->j',temp_2, r2, optimize=True)


                # ADC(3) ajk - i block
                
                temp =  0.5*np.einsum('xyde,deai->axyi',t2_1, v2e_so_vvvo)
                temp = temp[:,ij_ind[0],ij_ind[1],:]
                
                temp_1 = np.einsum('xlad,ydil->axyi',t2_1, v2e_so_ovoo)
                temp_2 = temp_1.transpose(0,2,1,3).copy()
                
                temp_1 = temp_1[:,ij_ind[0],ij_ind[1],:]
                temp_2 = temp_2[:,ij_ind[0],ij_ind[1],:]

                s2 += np.einsum('api,i->ap',temp, r1 ,optimize = True).reshape(-1)
                s2 += np.einsum('api,i->ap',temp_1, r1 ,optimize = True).reshape(-1)
                s2 -= np.einsum('api,i->ap',temp_2, r1 ,optimize = True).reshape(-1)
        
        
                #s = np.concatenate((s1,s2))
        return s1_a
    return sigma_



def calculate_GF(direct_adc,apply_H,omega,orb,T):


    broadening = direct_adc.broadening
    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    dim = nocc_a + nocc_b
    singles = nocc_a
    doubles =  nvir_a * nocc_a *nocc_a
    dim = singles + doubles

    iomega = omega + broadening*1j
    
    #imag_r = -(np.real(T[:nocc_a]))/broadening
    
    #print (np.linalg.norm(imag_r))

    r = np.zeros((doubles))
    for i in range(doubles):
        r[i] = 1

    
    sigma_a = apply_H(r)  

    print (sigma_a)
    exit()

    #sigma = apply_H(imag_r) 
    
    



    real_r =  (-omega*imag_r  + np.real(sigma))/broadening

    n_singles = nocc_so
    n_doubles = nocc_so * (nocc_so - 1) * nvir_so//2


    dim = n_singles + n_doubles


    z = np.zeros((dim))
    new_r = np.array(z,dtype = complex)
    new_r.imag = z

    new_r.real = real_r.copy()
    new_r.imag = imag_r.copy()

    new_r = solve_conjugate_gradients(direct_adc,apply_H,T,new_r,omega)

    gf = np.dot(T,new_r)

    return gf




def solve_conjugate_gradients(direct_adc,apply_H,T,r,omega,orb):
    
    maxiter = direct_adc.maxiter

    iomega = omega + direct_adc.broadening*1j
    
    # Compute residual
    
    omega_H = iomega*r - apply_H(r)
    res = T - omega_H

    rms = np.linalg.norm(res)/np.sqrt(res.size)

    if rms < 1e-8:
        return r

    d = res

    delta_new = np.dot(np.ravel(res), np.ravel(res))

    conv = False

    for imacro in range(maxiter):

        
        Ad = iomega*d - apply_H(d)

        # q = A * d
        q = Ad

        # alpha = delta_new / d . q
        alpha = delta_new / (np.dot(np.ravel(d), np.ravel(q)))

        # x = x + alpha * d
        r = r + alpha * d
 
        res = res - alpha * q 

        delta_old = delta_new
        delta_new = np.dot(np.ravel(res), np.ravel(res))

        beta = delta_new/delta_old

        d = res + beta * d

        # Compute RMS of the residual
        rms = np.linalg.norm(res)/np.sqrt(res.size)
        
        print ("freq","   ",iomega,"   ","Iteration ", imacro, ": RMS = %8.4e" % rms)

        if abs(rms) < direct_adc.tol:
            conv = True
            break

    if conv:
        print ("Iterations converged")
    else:
        raise Exception("Iterations did not converge")

    return r
