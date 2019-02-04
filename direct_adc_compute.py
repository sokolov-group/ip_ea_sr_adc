import sys
import numpy as np
import time
from functools import reduce

def kernel(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10,threshold=100000, suppress=True)

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
    
    print ("Number of states:            ", direct_adc.nstates)
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
     
    print ("MP2 correlation energy:  ", e_mp2)
    print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")

    # Compute the sigma vector,preconditioner and guess vector
    apply_H, precond = define_H(direct_adc,t_amp)

    # Compute Green's functions directly
    dos = calc_density_of_states(direct_adc, apply_H,precond,t_amp)

    np.savetxt('density_of_states.txt', dos, fmt='%.8f')

    #print ("Computation successfully finished")
    #print ("Total time:", (time.time() - t_start, "sec"))

    # Save to a file or plot depeneding on user input
    # Plot     
   
    
def conventional(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10,threshold=100000, suppress=True)

    print ("\nStarting spin-orbital direct ADC code..\n")
    print ("Number of electrons:               ", direct_adc.nelec)
    print ("Number of alpha electrons:         ", direct_adc.nelec_a)
    print ("Number of beta electrons:          ", direct_adc.nelec_b)
    print ("Number of basis functions:         ", direct_adc.nmo)
    print ("Number of alpha occupied orbitals: ", direct_adc.nocc_a)
    print ("Number of beta occupied orbitals:  ", direct_adc.nocc_b)
    print ("Number of alpha virtual orbitals:  ", direct_adc.nvir_a)
    print ("Number of beta virtual orbitals:   ", direct_adc.nvir_b)
    print ("Number of states:                  ", direct_adc.nstates)
    print ("Nuclear repulsion energy:          ", direct_adc.enuc,"\n")
    
    print ("SCF orbital energies(alpha):\n", direct_adc.mo_energy_a, "\n")
    print ("SCF orbital energies(beta):\n", direct_adc.mo_energy_b, "\n")

    t_start = time.time()

    # Compute amplitudes
    t_amp = compute_amplitudes(direct_adc)
    
    # Compute MP2 energy
    e_mp2 = compute_mp2_energy(direct_adc, t_amp)
     
    print ("MP2 correlation energy:  ", e_mp2)
    print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")

    # Compute the sigma vector,preconditioner and guess vector
    apply_H, precond, x0 = setup_davidsdon(direct_adc, t_amp)

    E, U = direct_adc.davidson(apply_H, x0, precond, nroots = direct_adc.nstates, verbose = 6, max_cycle=150, max_space=12)

    print ("\n%s excitation energies (a.u.):" % (direct_adc.method))
    print (E.reshape(-1, 1))
    print ("\n%s excitation energies (eV):" % (direct_adc.method))
    E_ev = E * 27.2114
    print (E_ev.reshape(-1, 1))

    # Compute Green's functions directly
#    dos = calc_density_of_states(direct_adc, apply_H, t_amp)

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
        
        t1_3_b = np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b,t1_2_b)
        t1_3_b += np.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_a)
        
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a,t1_2_a)
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_b)
        
        
        t1_3_b -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b,t1_2_b)
        t1_3_b -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_a)

        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a)
        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_b)
        
        
        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b,t1_2_b)
        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_a)
        

        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a,t1_2_a)
        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_b)
        
        
        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b,t1_2_b)
        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_a)



        t1_3_a += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_a)
        t1_3_a += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_ab)
        
        
        t1_3_b += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_b)
        t1_3_b += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_ab)


        t1_3_a += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_a)
        t1_3_a += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_ab)


        t1_3_b += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_b)
        t1_3_b += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_ab)


        t1_3_a -= 0.5*np.einsum('lmad,lmid->ia',t2_2_a,v2e_ooov_a)
        t1_3_a -= np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab)
        

        t1_3_b -= 0.5*np.einsum('lmad,lmid->ia',t2_2_b,v2e_ooov_b)
        t1_3_b -= np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab)

        t1_3_a += 0.5*np.einsum('ilde,alde->ia',t2_2_a,v2e_vovv_a)
        t1_3_a += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab)
        

        t1_3_b += 0.5*np.einsum('ilde,alde->ia',t2_2_b,v2e_vovv_b)
        t1_3_b += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab)


        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a)
        t1_3_a += np.einsum('ilfd,aefm,mled->ia',t2_1_ab,v2e_vvvo_a,t2_1_ab)
        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab)
        t1_3_a += np.einsum('ilfd,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_b)
        t1_3_a -= np.einsum('ildf,aemf,mlde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)

        t1_3_b -= np.einsum('ildf,aefm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b)
        t1_3_b += np.einsum('ilfd,aefm,mled->ia',t2_1_ab,v2e_vvvo_b,t2_1_ab)
        t1_3_b -= np.einsum('ildf,aefm,lmde->ia',t2_1_b,v2e_vvvo_ab,t2_1_ab)
        t1_3_b += np.einsum('ilfd,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_b)
        t1_3_b -= np.einsum('ildf,aemf,mlde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        


        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a)
        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_b)
        t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab)
        t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab)
        

        t1_3_b += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b)
        t1_3_b += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_a,t2_1_a)
        t1_3_b += np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab)
        t1_3_b += np.einsum('ilaf,defm,lmde->ia',t2_1_b,v2e_vvvo_ab,t2_1_ab)

        t1_3_a += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        t1_3_a += np.einsum('inde,anlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        

        t1_3_b += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b)
        t1_3_b += np.einsum('inde,anlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)

        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        t1_3_a -= 0.5 * np.einsum('inad,neml,mlde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab)
        t1_3_a -= 0.5 * np.einsum('inad,nelm,lmde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab)
        t1_3_a -= 0.5 *np.einsum('inad,enlm,lmed->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_a -= 0.5*np.einsum('inad,enml,mled->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_b)
        
        t1_3_b += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b)
        t1_3_b -= 0.5 * np.einsum('inad,neml,mlde->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab)
        t1_3_b -= 0.5 * np.einsum('inad,nelm,lmde->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab)
        t1_3_b -= 0.5 *np.einsum('inad,enlm,lmed->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_b -= 0.5*np.einsum('inad,enml,mled->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_b += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_a,t2_1_a)

        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_a,t2_1_ab)
        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_ab,t2_1_b)
        t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)


        t1_3_b -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b)
        t1_3_b -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_ab)
        t1_3_b -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_a,v2e_vooo_ab,t2_1_a)
        t1_3_b -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)



        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_a,v2e_vvov_a,t2_1_a)
        t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_a,t2_1_ab)
        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_ab,t2_1_b)
        t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        
        t1_3_b += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_b,t2_1_b)
        t1_3_b += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_b,t2_1_ab)
        t1_3_b += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_a,v2e_vvov_ab,t2_1_a)
        t1_3_b += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        t1_3_a -= np.einsum('lnde,emin,lmad->ia',t2_1_a,v2e_vooo_a,t2_1_a)
        t1_3_a += np.einsum('lnde,mein,lmad->ia',t2_1_ab,v2e_ovoo_ab,t2_1_a)
        t1_3_a += np.einsum('nled,emin,mlad->ia',t2_1_ab,v2e_vooo_a,t2_1_ab)
        t1_3_a += np.einsum('lned,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_a -= np.einsum('lnde,mein,mlad->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab)

        
        t1_3_b -= np.einsum('lnde,emin,lmad->ia',t2_1_b,v2e_vooo_b,t2_1_b)
        t1_3_b += np.einsum('lnde,mein,lmad->ia',t2_1_ab,v2e_ovoo_ab,t2_1_b)
        t1_3_b += np.einsum('nled,emin,mlad->ia',t2_1_ab,v2e_vooo_b,t2_1_ab)
        t1_3_b += np.einsum('lned,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab)
        t1_3_b -= np.einsum('lnde,mein,mlad->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab)

        t1_3_a -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_a,v2e_vvov_a,t2_1_a)
        t1_3_a -= np.einsum('lmef,efid,lmad->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        t1_3_b -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_b,v2e_vvov_b,t2_1_b)
        t1_3_b -= np.einsum('lmef,efid,lmad->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab)
        
        
        
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
  
    return e_mp2


def calc_density_of_states(direct_adc,apply_H,precond,t_amp):

	nmo = direct_adc.nmo
	freq_range = direct_adc.freq_range	
	broadening = direct_adc.broadening
	step = direct_adc.step

	freq_range = np.arange(freq_range[0],freq_range[1],step)


	k = np.zeros((nmo,nmo))
	gf_a = np.array(k,dtype = complex)
	gf_a.imag = k

	gf_b = np.array(k,dtype = complex)
	gf_b.imag = k
	
	gf_a_trace= []
	gf_b_trace= []
	gf_im_trace = [] 
	
	for freq in freq_range:
       
		omega = freq
		iomega = freq + broadening*1j


		for orb in range(nmo):

			T_a = calculate_T(direct_adc, t_amp, orb, spin = "alpha")
			gf_a[orb,orb] = calculate_GF(direct_adc,apply_H,precond,omega,orb,T_a)
			
			T_b = calculate_T(direct_adc, t_amp, orb, spin = "beta")
			gf_b[orb,orb] = calculate_GF(direct_adc,apply_H,precond,omega,orb,T_b)

		gf_a_trace = -(1/(np.pi))*np.trace(gf_a.imag)
		gf_b_trace = -(1/(np.pi))*np.trace(gf_b.imag)
		gf_trace = np.sum([gf_a_trace,gf_b_trace])
		gf_im_trace.append(gf_trace)
        

	return gf_im_trace         

####	freq = -0.40
####	omega = freq
####	iomega = freq + broadening*1j
####	
####	orb = 5
####	
####	T_a = calculate_T(direct_adc, t_amp, orb, spin = "beta")
####	gf_a[orb,orb] = calculate_GF(direct_adc,apply_H,omega,orb,T_a)
####	
####	gf_a_trace = -(1/(np.pi))*np.trace(gf_a.imag)
####	gf_im_trace.append(gf_a_trace)
####	
####	return gf_im_trace








def calculate_T(direct_adc, t_amp, orb, spin=None):

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
    n_doubles_bab = nvir_b * nocc_a* nocc_b  
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2
    
    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

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
    
    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    



    T = np.zeros((dim))
    
    
    #ADC(2) 1h part

    if spin=="alpha":
   
        if orb < nocc_a:
            
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
        
        else :    

            T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]
        

            #ADC(2) 2h-1p part

            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
            t2_1_t_a = t2_1_t.transpose(2,1,0)           
            
            t2_1_t_ab = -t2_1_ab.transpose(3,2,0,1)           

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
            

        if(method=='adc(2)-e'or method=='adc(3)'):
        
       
            t2_2_a, t2_2_ab, t2_2_b = t2_2
   
            #ADC(3) 2h-1p part 
        
   
            if orb >= nocc_a:
       
   
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
                t2_2_t_a = t2_2_t.transpose(2,1,0)           
                
                t2_2_t_ab = -t2_2_ab.transpose(3,2,0,1)           
   
                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         

            #ADC(3) 1h part 

            if orb < nocc_a:
 
                T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
        

                T[s_a:f_a] += 0.25*np.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 

            else: 
    
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc_a),:], t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_ab[:,:,(orb-nocc_a),:], t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]


        
    if spin=="beta":
   
        if orb < nocc_b:
            
            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_1_b, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
       
        else :    

            T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]
        

            #ADC(2) 2h-1p part



            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
            t2_1_t_b = t2_1_t.transpose(2,1,0)           

            t2_1_t_ab = -t2_1_ab.transpose(3,2,0,1)           

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[(orb-nocc_b),:,:,:].reshape(-1)


        if(method=='adc(2)-e'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = t2_2

            if orb >= nocc_b:


                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
                t2_2_t_b = t2_2_t.transpose(2,1,0)           

                t2_2_t_ab = -t2_2_ab.transpose(3,2,0,1)           

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[(orb-nocc_b),:,:,:].reshape(-1)
                   


        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         
            t2_2_a, t2_2_ab, t2_2_b = t2_2

            #ADC(3) 1h part 

            if orb < nocc_b:
 
                T[s_b:f_b] += 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
        

                T[s_b:f_b] += 0.25*np.einsum('ikdc,kdc->i',t2_1_b, t2_2_b[:,orb,:,:],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 

            else: 
    
                T[s_b:f_b] += 0.5*np.einsum('ikc,kc->i',t2_1_b[:,:,(orb-nocc_b),:], t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*np.einsum('ikc,kc->i',t2_1_ab[:,:,(orb-nocc_b),:], t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]
        

    return T


def get_Mij(direct_adc,t_amp):


    method = direct_adc.method
   

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    #t2_2_a, t2_2_ab, t2_2_b = t2_2
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
    M_ij_a = np.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
    M_ij_b = np.einsum('ij,j->ij', idn_occ_b ,e_occ_b)



    # Second-order terms

    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a)
    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab)
    
    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b)
    M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)


    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)


    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)


    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)


    M_ij_a += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_a, v2e_oovv_a)
    M_ij_a += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_b += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_b, v2e_oovv_b)
    M_ij_b += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_a += 0.5 *  np.einsum('jlde,deil->ij',t2_1_a, v2e_vvoo_a)
    M_ij_a += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)

    M_ij_b += 0.5 *  np.einsum('jlde,deil->ij',t2_1_b, v2e_vvoo_b)
    M_ij_b += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)

    # Third-order terms
    

    if (method == "adc(3)"):

        t2_2_a, t2_2_ab, t2_2_b = t2_2
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
    t1_2_a, t1_2_b = t1_2
    

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
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = direct_adc.v2e.ovvv
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b = direct_adc.v2e.vvov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov

    v2e_vooo_1_a = v2e_vooo_a[:,:,ij_ind_a[0],ij_ind_a[1]].transpose(1,0,2).reshape(nocc_a,-1)
    v2e_vooo_1_b = v2e_vooo_a[:,:,ij_ind_b[0],ij_ind_b[1]].transpose(1,0,2).reshape(nocc_a,-1)

    v2e_vooo_1_ab_a = -v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_a, -1)
    v2e_vooo_1_ab_b = -v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_b, -1)

    v2e_oovo_1_a = v2e_oovo_a[ij_ind_a[0],ij_ind_a[1],:,:].transpose(1,0,2)
    v2e_oovo_1_b = v2e_oovo_b[ij_ind_b[0],ij_ind_b[1],:,:].transpose(1,0,2)
    v2e_oovo_1_ab = -v2e_oovo_ab.transpose(2,0,1,3)

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_a.reshape((nvir_b,nocc_b,nocc_b))
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
    
    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_aij_bab = D_n_bab.reshape(-1)

    
    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_aij_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    
    
    M_ij_a, M_ij_b = get_Mij(direct_adc,t_amp)
    
    precond = np.zeros(dim)
    
    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    precond[s_a:f_a] = M_ij_a_diag.copy()
    precond[s_b:f_b] = M_ij_b_diag.copy()
    
    # Compute precond in 2p1h-2p1h block
  
    precond[s_aaa:f_aaa] = D_aij_a
    precond[s_bab:f_bab] = D_aij_bab
    precond[s_aba:f_aba] = D_aij_aba
    precond[s_bbb:f_bbb] = D_aij_b

    def sigma_(r):

       
        z = np.zeros((dim))
        s = np.array(z,dtype = complex)
        s.imag = z

        
        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)


        # ADC(2) ij block
        
        s[s_a:f_a] = np.einsum('ij,j->i',M_ij_a,r_a) 
        s[s_b:f_b] = np.einsum('ij,j->i',M_ij_b,r_b) 
            
        # ADC(2) i - kja block
        
        
        s[s_a:f_a] += np.einsum('ip,p->i', v2e_vooo_1_a, r_aaa, optimize = True)
        s[s_a:f_a] -= np.einsum('iajk,akj->i', v2e_ovoo_ab, r_bab, optimize = True)



        s[s_b:f_b] += np.einsum('ip,p->i', v2e_vooo_1_b, r_bbb, optimize = True)
        s[s_b:f_b] += np.einsum('ip,p->i', v2e_vooo_1_ab_b, r_aba, optimize = True)



        # ADC(2) ajk - i block
        

        temp = np.einsum('jkai,i->ajk', v2e_oovo_a, r_a, optimize = True)
        s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)
        
        s[s_bab:f_bab] += np.einsum('ajki,i->ajk', v2e_oovo_1_ab, r_a, optimize = True).reshape(-1)

        s[s_aba:f_aba] += np.einsum('ajki,i->ajk', v2e_oovo_1_ab, r_b, optimize = True).reshape(-1)
        s[s_bbb:f_bbb] += np.einsum('api,i->ap', v2e_oovo_1_b, r_b, optimize = True).reshape(-1)


       
        # ADC(2) ajk - bil block

       

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb



        if (method == "adc(2)-e" or method == "adc(3)"):
       	
      
               t2_2_a, t2_2_ab, t2_2_b = t2_2

               #print("Calculating additional terms for adc(2)-e")	
       
               # ajk - bil block


               temp = s[s_bab:f_bab].reshape(nvir_b,nocc_a,nocc_b).transpose(0,2,1).copy()
               s[s_bab:f_bab] = temp.reshape(-1)

               temp = s[s_aba:f_aba].reshape(nvir_a,nocc_b,nocc_a).transpose(0,2,1).copy()
               s[s_aba:f_aba] = temp.reshape(-1)


               r_aaa = r_aaa.reshape(nvir_a,-1)
               r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)
               r_aba = r_aba.reshape(nvir_a,nocc_b,nocc_a)
               r_bbb = r_bbb.reshape(nvir_b,-1)

               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a),dtype=complex)
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()    
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()   

               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b),dtype=complex)
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()    
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()   


               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_a,r_aaa_u ,optimize = True)


               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)


               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_b,r_bbb_u,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)




               s[s_bab:f_bab] -= 0.5*np.einsum('jkli,ali->akj',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('jkil,ail->akj',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)


               s[s_aba:f_aba] -= 0.5*np.einsum('jkli,ali->akj',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jkil,ail->akj',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)



               temp = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -=0.5* np.einsum('bkla,blj->ajk',v2e_voov_ab,r_bab,optimize = True)
             
  
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             

               s[s_bab:f_bab] -= 0.5*np.einsum('bkla,bjl->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('bkal,blj->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)
            



               temp_1 = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp_1 -= 0.5*np.einsum('bkla,blj->ajk',v2e_voov_ab,r_aba,optimize = True)


               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)    


               s[s_aba:f_aba] -= 0.5*np.einsum('bkla,bjl->ajk',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('bkal,blj->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)


               temp = -0.5*np.einsum('bkla,bjl->ajk',v2e_voov_ab,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('bkal,blj->ajk',v2e_vovo_a,r_aba,optimize = True)



               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jbal,blk->ajk',v2e_ovvo_ab,r_bab,optimize = True)


               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             


               s[s_bab:f_bab] +=  0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)


               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('bjla,blk->ajk',v2e_voov_ab,r_aba,optimize = True)


               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)             


               s[s_aba:f_aba] += 0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)




               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('kbai,bij->ajk',v2e_ovvo_ab,r_bab,optimize = True)


               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             


               s[s_bab:f_bab] += 0.5*np.einsum('bkia,bij->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('bkai,bij->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)


               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('kbai,bij->ajk',v2e_ovvo_ab,r_aba,optimize = True)


               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)             


               s[s_aba:f_aba] += 0.5*np.einsum('bkia,bij->ajk',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('bkai,bij->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
                

               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_bab,optimize = True)



               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             


               s[s_bab:f_bab] += 0.5*np.einsum('bjai,bki->ajk',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)


               s[s_aba:f_aba] += 0.5*np.einsum('bjai,bki->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)


               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_aba,optimize = True)


               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)   
                

               temp = s[s_bab:f_bab].reshape(nvir_b,nocc_a,nocc_b).transpose(0,2,1).copy()
               s[s_bab:f_bab] = temp.reshape(-1)

               temp = s[s_aba:f_aba].reshape(nvir_a,nocc_b,nocc_a).transpose(0,2,1).copy()
               s[s_aba:f_aba] = temp.reshape(-1)


        if (method == "adc(3)"):
      
            #print("Calculating additional terms for adc(3)")	
             
               
               temp = s[s_bab:f_bab].reshape(nvir_b,nocc_a,nocc_b).transpose(0,2,1).copy()
               s[s_bab:f_bab] = temp.reshape(-1)
           
               temp = s[s_aba:f_aba].reshape(nvir_a,nocc_b,nocc_a).transpose(0,2,1).copy()
               s[s_aba:f_aba] = temp.reshape(-1)

               r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)
               r_aba = r_aba.reshape(nvir_a,nocc_b,nocc_a)

               # ADC(3) i - kja block

               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               temp = np.einsum('pbc,bcai->pai',t2_1_a_t,v2e_vvvo_a)
          
               r_aaa = r_aaa.reshape(nvir_a,-1)
               s[s_a:f_a] += np.einsum('pai,ap->i',temp, r_aaa, optimize=True)
               

               temp_1 = np.einsum('jkbc,ajk->abc',t2_1_ab,r_bab)

               s[s_a:f_a] -= np.einsum('abc,bcia->i',temp_1, v2e_vvov_ab, optimize=True)


               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               temp = np.einsum('pbc,bcai->pai',t2_1_b_t,v2e_vvvo_b)
          
               r_bbb = r_bbb.reshape(nvir_b,-1)
               s[s_b:f_b] += np.einsum('pai,ap->i',temp, r_bbb, optimize=True)
               

               temp_1 = np.einsum('jkbc,ajk->abc',t2_1_ab,r_aba)

               s[s_b:f_b] -= np.einsum('abc,bcia->i',temp_1, v2e_vvov_ab, optimize=True)
               


               r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a),dtype=complex)
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()    
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()    


               r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b),dtype=complex)
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()    
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()    


               r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)
               r_aba = r_aba.reshape(nvir_a,nocc_b,nocc_a)

               temp = np.zeros_like(r_bab,dtype=complex)
               temp = np.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
               temp -= np.einsum('jlab,ajk->blk',t2_1_ab,r_bab,optimize=True)


               temp_1 = np.zeros_like(r_bab,dtype=complex)
               temp_1 = np.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 -= np.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)
               

               temp_2 = np.einsum('jlba,akj->blk',t2_1_ab,r_bab)
               
               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blk,ilbk->i',temp_2,v2e_oovo_ab,optimize=True)


               temp = np.zeros_like(r_aba,dtype=complex)
               temp = np.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)
               temp -= np.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)


               temp_1 = np.zeros_like(r_aba,dtype=complex)
               temp_1 = np.einsum('jlab,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 -= np.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)
               

               temp_2 = np.einsum('jlba,akj->blk',t2_1_ab,r_aba,optimize=True)
               
               s[s_b:f_b] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blk,ilkb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blk,ilbk->i',temp_2,v2e_oovo_ab,optimize=True)

               temp = np.zeros_like(r_bab,dtype=complex)
               temp = -np.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               temp += np.einsum('klab,akj->blj',t2_1_ab,r_bab,optimize=True)
               
               temp_1 = np.zeros_like(r_bab,dtype=complex)
               temp_1 = -np.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blj,ilbj->i',temp_2,v2e_oovo_ab,optimize=True)


               temp = np.zeros_like(r_aba,dtype=complex)
               temp = -np.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)
               
               temp_1 = np.zeros_like(r_bab,dtype=complex)
               temp_1 = -np.einsum('klab,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('klba,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blj,iljb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blj,ilbj->i',temp_2,v2e_oovo_ab,optimize=True)

               # ADC(3) ajk - i block
                
                
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               temp = 0.5*np.einsum('pbc,bcai->api',t2_1_a_t,v2e_vvvo_a)
               
          
               s[s_aaa:f_aaa] += np.einsum('api,i->ap',temp, r_a, optimize=True).reshape(-1)



               temp_1 = -np.einsum('jkbc,bcia->iajk',t2_1_ab,v2e_vvov_ab)

               temp_1 = temp_1.reshape(nocc_a,-1)

               s[s_bab:f_bab] += np.einsum('ip,i->p',temp_1, r_a, optimize=True).reshape(-1)

    
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               temp = np.einsum('pbc,bcai->api',t2_1_b_t,v2e_vvvo_b)
          
               s[s_bbb:f_bbb] += 0.5*np.einsum('api,i->ap',temp, r_b, optimize=True).reshape(-1)

               temp_1 = -np.einsum('jkbc,bcia->iajk',t2_1_ab,v2e_vvov_ab)

               temp_1 = temp_1.reshape(nocc_a,-1)

               s[s_aba:f_aba] += np.einsum('ip,i->p',temp_1, r_b, optimize=True).reshape(-1)



               temp_1 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_ab)
              
               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_a)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_ab)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)
               

               temp  = -np.einsum('bkil,i->kbl',v2e_vooo_ab,r_a)
               temp_1  = -np.einsum('kbl,jlba->ajk',temp,t2_1_ab)


               s[s_bab:f_bab] += temp_1.reshape(-1)

               temp_1 = np.einsum('i,kbil->kbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,kbil->kbl',r_b, v2e_ovoo_ab)
              
               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_b)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_ab)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)
              

               temp  = -np.einsum('bkil,i->kbl',v2e_vooo_ab,r_b)
               temp_1  = -np.einsum('kbl,jlba->ajk',temp,t2_1_ab)


               s[s_aba:f_aba] += temp_1.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_ab)
              
               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_a)
               temp += np.einsum('jbl,klab->ajk',temp_2,t2_1_ab)

               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)
              
               temp  = np.einsum('jbl,klab->ajk',temp_2,t2_1_b)
               temp  += np.einsum('jbl,klab->ajk',temp_1,t2_1_ab)


               s[s_bab:f_bab] -= temp.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,jbil->jbl',r_b, v2e_ovoo_ab)
              
               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_b)
               temp += np.einsum('jbl,klab->ajk',temp_2,t2_1_ab)

               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)
              
               temp  = np.einsum('jbl,klab->ajk',temp_2,t2_1_a)
               temp  += np.einsum('jbl,klab->ajk',temp_1,t2_1_ab)


               s[s_aba:f_aba] -= temp.reshape(-1)
               
               temp = s[s_bab:f_bab].reshape(nvir_b,nocc_a,nocc_b).transpose(0,2,1)
               s[s_bab:f_bab] = temp.reshape(-1)


               temp = s[s_aba:f_aba].reshape(nvir_a,nocc_b,nocc_a).transpose(0,2,1)
               s[s_aba:f_aba] = temp.reshape(-1)

               s *= -1.0

        return s

    precond *= -1.0
    
    return sigma_, precond


def calculate_GF(direct_adc,apply_H,precond,omega,orb,T):


    method = direct_adc.method


    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb


    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    




    broadening = direct_adc.broadening
    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    iomega = omega + broadening*1j
    
    imag_r = -(np.real(T))/broadening

    sigma = apply_H(imag_r)

    real_r =  (-omega*imag_r  - np.real(sigma))/broadening


    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    z = np.zeros((dim))
    new_r = np.array(z,dtype = complex)
    new_r.imag = z

    new_r.real = real_r.copy()
    new_r.imag = imag_r.copy()
    

    new_r = solve_conjugate_gradients(direct_adc,apply_H,precond,T,new_r,omega,orb)
    

    gf = np.dot(T,new_r)

    return gf




def solve_conjugate_gradients(direct_adc,apply_H,precond,T,r,omega,orb):
    
    maxiter = direct_adc.maxiter

    iomega = omega + direct_adc.broadening*1j
     
    # Compute residual
    
    omega_H = iomega*r + apply_H(r)
    res = T - omega_H
    
     

    rms = np.linalg.norm(res)/np.sqrt(res.size)
    

    if rms < 1e-8:
        return r

    #d = (precond+1e-6)**(-1)*res
    d = res
    
    #delta_new = np.dot(np.ravel(res), np.ravel(res))
    delta_new = np.dot(np.ravel(res), np.ravel(d))

    conv = False

    for imacro in range(maxiter):

        
        Ad = iomega*d + apply_H(d)

        # q = A * d
        q = Ad

        # alpha = delta_new / d . q
        alpha = delta_new / (np.dot(np.ravel(d), np.ravel(q)))

        # x = x + alpha * d
        r = r + alpha * d
 
        res = res - alpha * q 
        s = (precond+1e-6)**(-1)*res
        #s = res

        delta_old = delta_new
        #delta_new = np.dot(np.ravel(res), np.ravel(res))
        delta_new = np.dot(np.ravel(res), np.ravel(s))

        beta = delta_new/delta_old

        d = res + beta * d
        #d = s + beta * d

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


def setup_davidsdon(direct_adc, t_amp):

    apply_H = None
    precond = None

    apply_H, precond = define_H(direct_adc, t_amp)

    x0 = compute_guess_vectors(direct_adc, precond)

    return apply_H, precond, x0


def compute_guess_vectors(direct_adc, precond, ascending = True):

    sort_ind = None
    if ascending:
        sort_ind = np.argsort(precond)
    else:
        sort_ind = np.argsort(precond)[::-1]

    x0s = np.zeros((precond.shape[0], direct_adc.nstates))
    min_shape = min(precond.shape[0], direct_adc.nstates)
    x0s[:min_shape,:min_shape] = np.identity(min_shape)

    x0 = np.zeros((precond.shape[0], direct_adc.nstates))
    x0[sort_ind] = x0s.copy()

    x0s = []
    for p in range(x0.shape[1]):
        x0s.append(x0[:,p])

    return x0s

