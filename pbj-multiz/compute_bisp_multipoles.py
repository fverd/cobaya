# This script computes the building blocks for the bispectrum multipoles,
# starting from the same parameter file used to run chains. As is, it only
# computes one multipole template at a time (so you need to run it with only
# one entry in the `Observables` key of the paramfile. The templates are
# directly saved in the `theory` folder, ready to be loaded by PBJ.
# Run the script as `python compute_bisp_multipoles.py paramfile` (takes ~2m)

import numpy as np
import sys
from math import *
import scipy.interpolate as spip
import numba as nb
import time

@nb.jit(nopython=True)
def Btree(SqQ1, SqQ2, SqQ3, PQ1, PQ2, PQ3, Q1, Q2, Q3, iQ1, iQ2, iQ3, q1z, q2z, res, obs):
    '''Computes the (unbinned) B_l building blocks
    '''
    P12 = PQ1*PQ2;  P23 = PQ2*PQ3;  P13 = PQ3*PQ1

    q3z = -q1z-q2z
    mu1 = q1z*iQ1;  mu2 = q2z*iQ2;  mu3 = q3z*iQ3
    if obs == 'B0': fact = 1.
    elif obs == 'B2': fact = 2.5*(3.*mu1**2 - 1.)
    elif obs == 'B4': fact = 1.125*(35.*mu1**4 - 30.*mu1**2 + 3.)


    mu12 = (SqQ3-SqQ1-SqQ2)*iQ1*iQ2*0.5
    mu23 = (SqQ1-SqQ2-SqQ3)*iQ2*iQ3*0.5
    mu13 = (SqQ2-SqQ3-SqQ1)*iQ3*iQ1*0.5

    S12 = mu12*mu12 - 1.;  S13 = mu13*mu13 - 1.;  S23 = mu23*mu23 - 1.

    F12 = 5/7. + mu12*(Q1*iQ2+Q2*iQ1)*0.5 + 2/7.*mu12*mu12
    F23 = 5/7. + mu23*(Q3*iQ2+Q2*iQ3)*0.5 + 2/7.*mu23*mu23
    F13 = 5/7. + mu13*(Q1*iQ3+Q3*iQ1)*0.5 + 2/7.*mu13*mu13

    G12 = 3/7. + mu12*(Q1*iQ2+Q2*iQ1)*0.5 + 4/7.*mu12*mu12
    G23 = 3/7. + mu23*(Q3*iQ2+Q2*iQ3)*0.5 + 4/7.*mu23*mu23
    G13 = 3/7. + mu13*(Q1*iQ3+Q3*iQ1)*0.5 + 4/7.*mu13*mu13

    A12 = 0.5*mu3*Q3*(mu1*iQ1 + mu2*iQ2)
    A23 = 0.5*mu1*Q1*(mu2*iQ2 + mu3*iQ3)
    A13 = 0.5*mu2*Q2*(mu1*iQ1 + mu3*iQ3)

    B12 = 0.5*mu3*Q3*(mu1*iQ1*mu2*mu2 + mu2*iQ2*mu1*mu1)
    B23 = 0.5*mu1*Q1*(mu2*iQ2*mu3*mu3 + mu3*iQ3*mu2*mu2)
    B13 = 0.5*mu2*Q2*(mu1*iQ1*mu3*mu3 + mu3*iQ3*mu1*mu1)

    # [b1*b1*b1] contribution
    res[0] = fact*(F12*P12 + F23*P23 + F13*P13)
    # b1*b1*b2 contribution
    res[1] = 0.5*fact*(P12 + P23 + P13)
    # [b1*b1*bG2] contribution
    res[2] = fact*(S12*P12 + S23*P23 + S13*P13)
    # [b1*b1*f] contribution
    res[3] = fact*(((mu1*mu1+mu2*mu2)*F12 + mu3*mu3*G12)*P12 +
                   ((mu2*mu2+mu3*mu3)*F23 + mu1*mu1*G23)*P23 +
                   ((mu1*mu1+mu3*mu3)*F13 + mu2*mu2*G13)*P13)
    # [b1*b1*b1*f] contribution
    res[4] = -fact*(A12*P12 + A23*P23 + A13*P13)
    # [f*f*b1*b1] contribution
    res[5] = -fact*(((mu1*mu1 + mu2*mu2)*A12 + B12)*P12 +
                    ((mu3*mu3 + mu2*mu2)*A23 + B23)*P23 +
                    ((mu1*mu1 + mu3*mu3)*A13 + B13)*P13)
    # [f*b1*b2] contribution
    res[6] = 0.5*fact*((mu1*mu1+mu2*mu2)*P12 +
                       (mu3*mu3+mu2*mu2)*P23 +
                       (mu1*mu1+mu3*mu3)*P13)
    # [f*b1*bG2] contribution
    res[7] = fact*((mu1*mu1+mu2*mu2)*S12*P12 +
                   (mu3*mu3+mu2*mu2)*S23*P23 +
                   (mu1*mu1+mu3*mu3)*S13*P13)
    # [b1*f*f] contribution
    res[8] = fact*((mu1*mu1*mu2*mu2*F12 + (mu1*mu1+mu2*mu2)*mu3*mu3*G12)*P12 +
                   (mu3*mu3*mu2*mu2*F23 + (mu3*mu3+mu2*mu2)*mu1*mu1*G23)*P23 +
                   (mu1*mu1*mu3*mu3*F13 + (mu1*mu1+mu3*mu3)*mu2*mu2*G13)*P13)
    # [b1*f*f*f] contribution
    res[9] = -fact*(((mu1*mu1+mu2*mu2)*B12 + mu1*mu1*mu2*mu2*A12 )*P12 +
                    ((mu3*mu3+mu2*mu2)*B23 + mu3*mu3*mu2*mu2*A23 )*P23 +
                    ((mu1*mu1+mu3*mu3)*B13 + mu1*mu1*mu3*mu3*A13 )*P13)
    # [b2*f*f] contribution
    res[10] = 0.5*fact*((mu1*mu1*mu2*mu2)*P12 +
                        (mu3*mu3*mu2*mu2)*P23 +
                        (mu1*mu1*mu3*mu3)*P13)
    # [bG2*f*f] contribution
    res[11] = fact*((mu1*mu1*mu2*mu2)*S12*P12 +
                    (mu3*mu3*mu2*mu2)*S23*P23 +
                    (mu1*mu1*mu3*mu3)*S13*P13)
    # [f*f*f] contribution
    res[12] = fact*mu1*mu1*mu2*mu2*mu3*mu3*(G12*P12 + G23*P23 + G13*P13)
    # [f*f*f*f]
    res[13] = -fact*(mu1*mu1*mu2*mu2*B12*P12 +
                     mu3*mu3*mu2*mu2*B23*P23 +
                     mu1*mu1*mu3*mu3*B13*P13)
    # contributions coming from SN
    res[14] = 0.5*fact*(PQ1 + PQ2 + PQ3)
    res[15] = fact*(mu1*mu1*PQ1 + mu2*mu2*PQ2 + mu3*mu3*PQ3)
    res[16] = 0.5*fact*(mu1*mu1*mu1*mu1*PQ1 +
                        mu2*mu2*mu2*mu2*PQ2 +
                        mu3*mu3*mu3*mu3*PQ3)


@nb.jit(nopython=True)
def numtri(dk, cf, Nb, iOpen):
    '''Counts the triangles
    '''
    N_T = 0
    ishift = int(cf/dk + 1.5 * iOpen)
    for i in nb.prange(1, Nb+1):
        for j in nb.prange(1, i+1):
            for l in nb.prange(max(1,i-j+1-ishift), j+1):
                N_T += 1
    return N_T


@nb.jit(nopython=True)
def makebins(dk, cf, Nb, iOpen, Ni, Nj, Nl, Bi, Bj, Bl, GetIdx, Symf):
    b = Nb+1
    ishift = int(cf/dk + 1.5 * iOpen)
    I = 1
    ni = 0
    nj = 0
    nl = 0
    for i in nb.prange(1, Nb+1):
        ni = cf+(i-1)*dk
        for j in nb.prange(1, i+1):
            nj = cf+(j-1)*dk
            for l in nb.prange(max(1,i-j+1-ishift), j+1):
                nl = cf+(l-1)*dk
                Ni[I] = ni
                Nj[I] = nj
                Nl[I] = nl
                Bi[I] = i
                Bj[I] = j
                Bl[I] = l
                GetIdx[i*b*b+j*b+l] = I
                flag = int(not(ni-nj)) + int(not(nj-nl))*2
                Symf[I] = (-5*flag*flag*flag + 24*flag*flag -37*flag + 36)/6
                I += 1


@nb.jit(nopython=True)
def binning(L, Nb, km2, kM2, Ptable, Sqrt, ISqrt, Bin, GetIdx, Symf, B_l, counts, obs):
    Res = np.zeros(17)
    b = Nb+1
    S1 = S2 = S3 = 0
    B1 = B2 = B3 = 0
    mult = 1
    I = 0
    norm1 = norm2 = norm3 = 0.
    inorm1 = inorm2 = inorm3 = 0.
    PQ1 = PQ2 = PQ3 = 0.
    symf = 0
    k1zmin = k1zmax = k2zmax = 0
    k3z = 0.
    for k1x in nb.prange(-L, L+1):
        for k1y in nb.prange(-L, L+1):
            if k1x*k1x+k1y*k1y <= kM2:
                for k2x in nb.prange(-L, L+1):
                    for k2y in nb.prange(-L, L+1):
                        if k2x*k2x+k2y*k2y <= kM2:
                            k1zmin = int(sqrt(max(0, km2-k1x*k1x-k1y*k1y)))
                            k1zmax = int(sqrt(kM2-k1x*k1x-k1y*k1y))+1
                            for k1z in nb.prange(-k1zmax, k1zmax+1):
                                S1 = k1x*k1x+k1y*k1y+k1z*k1z
                                if (km2 <= S1 <= kM2):
                                    PQ1 = Ptable[S1]
                                    inorm1 = ISqrt[S1]
                                    norm1 = Sqrt[S1]
                                    B1 = Bin[S1]
                                    mult = 1
                                    k2zmax = int(sqrt(kM2-k2x*k2x-k2y*k2y))
                                    for k2z in nb.prange(-k2zmax, k2zmax+1):
                                        S2 = k2x*k2x+k2y*k2y+k2z*k2z
                                        S3 = S1 + S2 + 2*(k1x*k2x + k1y*k2y + k1z*k2z)
                                        if (S2 >= km2) and (km2 <= S3 <= kM2):
                                            PQ2 = Ptable[S2]
                                            inorm2 = ISqrt[S2]
                                            PQ3 = Ptable[S3]
                                            inorm3 = ISqrt[S3]
                                            norm2 = Sqrt[S2]
                                            norm3 = Sqrt[S3]
                                            B2 = Bin[S2]
                                            B3 = Bin[S3]
                                            I = GetIdx[B1*b*b+B2*b+B3]
                                            if I:
                                                symf = Symf[I]*mult
                                                counts[I] += symf
                                                Btree(S1, S2, S3, PQ1, PQ2, PQ3, norm1, norm2, norm3, inorm1, inorm2, inorm3, k1z, k2z, Res, obs)
                                                for h in nb.prange(17):
                                                    B_l[h,I] += 2.0*Res[h]*symf

#-------------------------------------------------------------------------------
## Initialize pbj with the same parameter file used for the chains
print("Initializing PBJ...")
import PBJ
inputfile = sys.argv[1]
with open(inputfile) as f:
    Dict = {k: eval(v) for line in f for (k,v) in (line.strip().split(None,1),)}

pbj = PBJ.pbj(Dict)
pbj.initialise_grid()

Nbins = ceil(pbj.nbinsB)

if pbj.IRres == False:
    ir_res=''
    iPlin = spip.interp1d(np.log(pbj.kL), np.log(pbj.Dz*pbj.Dz*pbj.PL), kind='cubic')
    PkL = lambda x: np.exp(iPlin(np.log(x)))
else:
    ir_res='_IRres'
    iPlin = spip.interp1d(np.log(pbj.kL), np.log(pbj.PowerfuncIR_real()[0]), kind='cubic')
    PkL = lambda x: np.exp(iPlin(np.log(x)))

kf2 = pbj.kf*pbj.kf
Nmax = pbj.cf + (Nbins-1.)*pbj.dk
kmin = pbj.cf-0.5*pbj.dk
kmax = Nmax+0.5*pbj.dk
km2 = kmin*kmin
kM2 = kmax*kmax
L = int(floor(kmax))
L2 = int(kM2)

Ptable = np.zeros([L2+1])
Sqrt = np.zeros([L2+1])
ISqrt = np.zeros([L2+1])
Bin = np.zeros([L2+1], dtype=int)

print("Computing P_table")
for i in range(1, L2+1):
    Sqrt[i] = sqrt(float(i))
    ISqrt[i] = 1./Sqrt[i]
    Bin[i] = int((Sqrt[i]-pbj.cf)/pbj.dk + 1.5)
    Ptable[i] = PkL(pbj.kf*Sqrt[i])

print("Counting triangles and making bins...")
N_T = numtri(pbj.dk, pbj.cf, Nbins, 1)
Ni = np.zeros([N_T+1])
Nj = np.zeros([N_T+1])
Nl = np.zeros([N_T+1])
Bi = np.zeros([N_T+1], dtype=int)
Bj = np.zeros([N_T+1], dtype=int)
Bl = np.zeros([N_T+1], dtype=int)
Symf = np.zeros([N_T+1], dtype=int)

GetIdx = np.zeros([(Nbins+1)**3], dtype=int)

makebins(pbj.dk, pbj.cf, Nbins, 1, Ni, Nj, Nl, Bi, Bj, Bl, GetIdx, Symf)

counts = np.zeros([N_T+1], dtype=int)
B_l = np.zeros([17,N_T+1])

print("Binning!")
start = time.time()
binning(L, Nbins, km2, kM2, Ptable, Sqrt, ISqrt, Bin, GetIdx, Symf, B_l, counts,
        Dict['Observables'][0])
end = time.time()
print(end-start)

B_l = B_l[:,1:] / counts[None,1:]

np.savetxt('theory/'+Dict['Observables'][0]+'_template_Minerva_cut42_s'+\
           str(pbj.dk)+'_c'+str(pbj.cf)+'_n'+str(Nbins)+ir_res+'.dat', B_l.T)

