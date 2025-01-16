import numpy as np
from numpy import pi
# Functions for bispectrum kernels and derivatives w.r.t. k_i, i=[1,2,3]
def muij(K):
    ki, kj, kl = K
    return 0.5*(kl**2 - ki**2 - kj**2)/(ki*kj)

def F_2(ki,kj,kl):
    return (5/7. + 0.5*muij([ki,kj,kl]) * (ki/kj + kj/ki) +
            2/7. * muij([ki,kj,kl])**2)

def G_2(ki,kj,kl):
    return (3/7. + 0.5*muij([ki,kj,kl]) * (ki/kj + kj/ki) +
            4/7. * muij([ki,kj,kl])**2)

def F_2_dlnk1(ki,kj,kl):
    out = -0.5 - 0.5*ki**2/kj**2 - 4./7.0*ki*muij([ki, kj, kl])/kj - \
        kj*muij([ki, kj, kl])/ki - 4./7.0*muij([ki,kj,kl])**2
    return out

def F_2_dlnk2(ki,kj,kl):
    out = -0.5 - 0.5*kj**2/ki**2 - ki*muij([ki, kj, kl])/kj - \
        4./7.0*kj*muij([ki, kj, kl])/ki - 4.0/7.0*muij([ki, kj, kl])**2
    return out

def F_2_dlnk3(ki,kj,kl):
    out = 0.5 * kl**2 * (ki**2 + kj**2 + 8./7.0*ki*kj*muij([ki, kj, kl])) /\
        (ki**2*kj**2)
    return out

def G_2_dlnk1(ki,kj,kl):
    out = -0.5 - 0.5*ki**2/kj**2 - 8.0/7.0*ki*muij([ki, kj, kl])/kj - \
        kj*muij([ki, kj, kl])/ki - 8.0/7.0*muij([ki, kj, kl])**2
    return out

def G_2_dlnk2(ki,kj,kl):
    out = -0.5 - 0.5*kj**2/ki**2 - ki*muij([ki, kj, kl])/kj - \
        8./7.0*kj*muij([ki, kj, kl])/ki - 8.0/7.0*muij([ki, kj, kl])**2
    return out

def G_2_dlnk3(ki,kj,kl):
    out = 0.5*kl**2*(ki**2 + kj**2 + 16./7.0*ki*kj*muij([ki, kj, kl])) /\
        (ki**2*kj**2)
    return out

def S_2_dlnk1(ki,kj,kl):
    out = -2.0*muij([ki, kj, kl])*(ki + kj*muij([ki, kj, kl]))/kj
    return out

def S_2_dlnk2(ki,kj,kl):
    out = -2.0*muij([ki, kj, kl])*(kj + ki*muij([ki, kj, kl]))/ki
    return out

def S_2_dlnk3(ki,kj,kl):
    out = 2.0*muij([ki, kj, kl])*kl**2/(ki*kj)
    return out


def Bl_templates_k(k1, k2, k3):

    K = {}
    mu12 = muij([k1, k2, k3])
    mu23 = muij([k2, k3, k1])
    mu31 = muij([k3, k1, k2])

    K['mu12'] = mu12
    K['mu23'] = mu23
    K['mu31'] = mu31

    K['F_12'] = F_2(k1, k2, k3)
    K['F_23'] = F_2(k2, k3, k1)
    K['F_31'] = F_2(k3, k1, k2)

    K['G_12'] = G_2(k1, k2, k3)
    K['G_23'] = G_2(k2, k3, k1)
    K['G_31'] = G_2(k3, k1, k2)

    K['S_12'] = mu12**2 - 1.
    K['S_23'] = mu23**2 - 1.
    K['S_31'] = mu31**2 - 1.

    K['dF_12_dlnk1'] = F_2_dlnk1(k1, k2, k3)
    K['dF_31_dlnk3'] = F_2_dlnk1(k3, k1, k2)
    K['dF_23_dlnk2'] = F_2_dlnk1(k2, k3, k1)

    K['dF_12_dlnk2'] = F_2_dlnk2(k1, k2, k3)
    K['dF_31_dlnk1'] = F_2_dlnk2(k3, k1, k2)
    K['dF_23_dlnk3'] = F_2_dlnk2(k2, k3, k1)

    K['dF_12_dlnk3'] = F_2_dlnk3(k1, k2, k3)
    K['dF_31_dlnk2'] = F_2_dlnk3(k3, k1, k2)
    K['dF_23_dlnk1'] = F_2_dlnk3(k2, k3, k1)

    K['dG_12_dlnk1'] = G_2_dlnk1(k1, k2, k3)
    K['dG_31_dlnk3'] = G_2_dlnk1(k3, k1, k2)
    K['dG_23_dlnk2'] = G_2_dlnk1(k2, k3, k1)

    K['dG_12_dlnk2'] = G_2_dlnk2(k1, k2, k3)
    K['dG_31_dlnk1'] = G_2_dlnk2(k3, k1, k2)
    K['dG_23_dlnk3'] = G_2_dlnk2(k2, k3, k1)

    K['dG_12_dlnk3'] = G_2_dlnk3(k1, k2, k3)
    K['dG_31_dlnk2'] = G_2_dlnk3(k3, k1, k2)
    K['dG_23_dlnk1'] = G_2_dlnk3(k2, k3, k1)


    K['dS_12_dlnk1'] = S_2_dlnk1(k1, k2, k3)
    K['dS_31_dlnk3'] = S_2_dlnk1(k3, k1, k2)
    K['dS_23_dlnk2'] = S_2_dlnk1(k2, k3, k1)

    K['dS_12_dlnk2'] = S_2_dlnk2(k1, k2, k3)
    K['dS_31_dlnk1'] = S_2_dlnk2(k3, k1, k2)
    K['dS_23_dlnk3'] = S_2_dlnk2(k2, k3, k1)

    K['dS_12_dlnk3'] = S_2_dlnk3(k1, k2, k3)
    K['dS_31_dlnk2'] = S_2_dlnk3(k3, k1, k2)
    K['dS_23_dlnk1'] = S_2_dlnk3(k2, k3, k1)

    k31 = k3/k1
    k32 = k3/k2
    K['k31'] = k31
    K['k32'] = k32
    K['dk31_dlnk1'] = -k31
    K['dk31_dlnk2'] = np.zeros(len(k31))
    K['dk31_dlnk3'] = k31
    K['dk32_dlnk1'] = np.zeros(len(k31))
    K['dk32_dlnk2'] = -k32
    K['dk32_dlnk3'] = k32

    k21 = k2/k1
    k23 = k2/k3
    K['k21'] = k21
    K['k23'] = k23
    K['dk21_dlnk1'] = -k21
    K['dk21_dlnk2'] = k21
    K['dk21_dlnk3'] = np.zeros(len(k21))
    K['dk23_dlnk1'] = np.zeros(len(k21))
    K['dk23_dlnk2'] = k23
    K['dk23_dlnk3'] = -k23


    k12 = k1/k2
    k13 = k1/k3
    K['k12'] = k12
    K['k13'] = k13
    K['dk12_dlnk1'] = k12
    K['dk12_dlnk2'] = -k12
    K['dk12_dlnk3'] = np.zeros(len(k12))
    K['dk13_dlnk1'] = k13
    K['dk13_dlnk2'] = np.zeros(len(k12))
    K['dk13_dlnk3'] = -k13

    F = [K['F_12'], K['F_31'], K['F_23']]
    G = [K['G_12'], K['G_31'], K['G_23']]
    S = [K['S_12'], K['S_31'], K['S_23']]
    dF = [(K['dF_12_dlnk1'], K['dF_12_dlnk2'], K['dF_12_dlnk3']),
        (K['dF_31_dlnk1'], K['dF_31_dlnk2'], K['dF_31_dlnk3']),
        (K['dF_23_dlnk1'], K['dF_23_dlnk2'], K['dF_23_dlnk3'])]
    dG = [(K['dG_12_dlnk1'], K['dG_12_dlnk2'], K['dG_12_dlnk3']),
        (K['dG_31_dlnk1'], K['dG_31_dlnk2'], K['dG_31_dlnk3']),
        (K['dG_23_dlnk1'], K['dG_23_dlnk2'], K['dG_23_dlnk3'])]
    dS = [(K['dS_12_dlnk1'], K['dS_12_dlnk2'], K['dS_12_dlnk3']),
        (K['dS_31_dlnk1'], K['dS_31_dlnk2'], K['dS_31_dlnk3']),
        (K['dS_23_dlnk1'], K['dS_23_dlnk2'], K['dS_23_dlnk3'])]
    KK = [ K['k31'], K['k32'], K['k21'], K['k23'], K['k12'], K['k13']]

    dKK = [ (K['dk31_dlnk1'], np.zeros(len(k31)), K['dk31_dlnk3']),
            (np.zeros(len(k31)), K['dk32_dlnk2'], K['dk32_dlnk3']),
            (K['dk21_dlnk1'], K['dk21_dlnk2'], np.zeros(len(k31))),
            (np.zeros(len(k31)), K['dk23_dlnk2'], K['dk23_dlnk3']),
            (K['dk12_dlnk1'], K['dk12_dlnk2'], np.zeros(len(k31))),
            (K['dk13_dlnk1'], np.zeros(len(k31)), K['dk13_dlnk3'])]

    return K, F, G, S, dF, dG, dS, KK, dKK


def Bl_templates_angles(a, k1, k2, k3):

    def Ia(bc, a, K):
        k1, k2, k3 = K
        if bc == '00':
            return ((2.*pi*(1.+(-1)**a)) / (1.+a))
        elif bc == '01':
            return ((2.*pi*(-1.+(-1)**a)*(k1 + k2*muij(K))) / ((2.+a)*k3))
        elif bc == '02':
            return ((2.*pi * (1.+(-1)**a) *
                     ((1.+a)*k1**2 + 2.*(1.+a)*k1*k2*muij(K) +
                      k2**2*(1.+a*muij(K)**2))) / ((1.+a)*(3.+a)*k3**2))
        elif bc == '03':
            return ((2.*pi * (-1.+(-1)**a) * (k1 +  k2*muij(K)) *
                     ((2.+a)*k1**2 + 2.*(2.+a)*k1*k2*muij(K) +
                      k2**2*(3.+(a-1.)*muij(K)**2))) / ((2.+a)*(4.+a)*k3**3))
        elif bc == '04':
            return ((2.*pi * (1.+(-1)**a) *
                     ((1.+a)*(3.+a)*k1**4 +
                      4*(1.+a)*(3.+a)*k1**3*k2*muij(K) +
                      4*(1.+a)*k1*k2**3*muij(K)*(3.+a*muij(K)**2) +
                      6*k1**2*k2**2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                      k2**4*(3.+6.*a*muij(K)**2+(a-2.)*a*muij(K)**4)))/
                    ((1.+a)*(3.+a)*(5.+a)*k3**4))
        elif bc == '05':
            return ((2.*pi*(-1.+(-1)**a)*(k1 + k2*muij(K))*
                     ((2.+a)*(4.+a)*k1**4+4.*(2.+a)*(4.+a)*k1**3*k2*muij(K) +
                      4.*(2.+a)*k1*k2**3*muij(K)*(5.+(-1.+a)*muij(K)**2) +
                      2.*(2.+a)*k1**2*k2**2*(5.+(7.+3*a)*muij(K)**2) +
                      k2**4*(15.+(-1.+a)*muij(K)**2*(10.+(-3.+a)*muij(K)**2))))/
                    ((2.+a)*(4.+a)*(6.+a)*k3**5))
        elif bc == '06':
            return ((2.*pi*(1.+(-1)**a)*
                     ((1.+a)*(3.+a)* (5.+a)*k1**6 +
                      6.*(1.+a)*(3.+a)* (5.+a)*k1**5*k2*muij(K) +
                      20.*(1.+a)*(3.+ a)*k1**3*k2**3*muij(K)*
                      (3.+(2.+a)*muij(K)**2) + 15.*(1.+a)*(3.+a)*k1**4*k2**2*
                      (1.+(4.+a)*muij(K)**2) + 6.*(1.+a)*k1*k2**5*muij(K)*
                      (15.+10.*a*muij(K)**2 +
                       (-2.+a)*a*muij(K)**4) + 15.*(1.+a)*k1**2*k2**4*
                      (3.+(2.+a)*muij(K)**2*(6.+a*muij(K)**2)) +
                      k2**6*(15.+a*muij(K)**2*(45.+(-2.+a)*muij(K)**2*
                                               (15.+(-4.+a)*muij(K)**2)))))/
                    ((1.+a)*(3.+a)*(5.+a)*(7.+a)*k3**6))

        elif bc == '10':
            return -((2.*pi * (-1.+(-1)**a) * muij(K)) / (2.+a))
        elif bc == '11':
            return -((2.*pi * (1.+(-1)**a)*
                      (k2+(1.+a)*k1*muij(K)+a*k2*muij(K)**2))/((1.+a)*(3.+a)*k3))
        elif bc == '12':
            return -((2.*pi * (-1.+(-1)**a) *
                      ((2.+a)*k1**2*muij(K) + k2**2*muij(K)*(3.+(a-1.)*muij(K)**2)+
                       2*k1*(k2+(1.+a)*k2*muij(K)**2)))/((2.+a)*(4.+a)*k3**2))
        elif bc == '13':
            return -((2.*pi * (1.+(-1)**a) *
                      ((1.+a)*(3.+a)*k1**3*muij(K) +
                       3.*(1.+a)*k1*k2**2*muij(K)*(3.+a*muij(K)**2)+
                       3.*k1**2*k2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                       k2**3*(3.+6.*a*muij(K)**2+(a-2.)*a*
                              muij(K)**4))) / ((1.+a)*(3.+a)*(5.+a)*k3**3))
        elif bc == '14':
            return -((2.*pi * (-1.+(-1)**a) *
                      ((2.+a)*(4.+a)*k1**4*muij(K) +
                       6.*(2.+a)*k1**2*k2**2*muij(K)*
                       (3.+(1.+a)*muij(K)**2)+
                       4.*k1**3*k2*(2.+a+(2.+a)*(3.+a)*muij(K)**2) +
                       4.*k1*k2**3*(3.+6.*(1.+a)*muij(K)**2+
                                    (-1.+a**2)*muij(K)**4) +
                       k2**4*muij(K)*(15.+(-1.+a)*muij(K)**2*
                                      (10.+(-3.+a)*muij(K)**2)))) /
                     ((2.+a)*(4.+a)*(6.+a)*k3**4))
        elif bc == '15':
            return ((-2.*pi*(1. + (-1)**a)*
                     ((1. + a)*(3. + a)*(5. + a)*k1**5*muij(K) +
                      10.*(1 + a)*(3 + a)*k1**3*k2**2*muij(K)*
                      (3. + (2 + a)*muij(K)**2) +
                      5.*(1 + a)*(3 + a)*k1**4*k2*(1 + (4 + a)*muij(K)**2) +
                      5.*(1 + a)*k1*k2**4*muij(K)*
                      (15. + 10*a*muij(K)**2 + (-2 + a)*a*muij(K)**4) +
                      10.*(1. + a)*k1**2*k2**3*
                      (3. + (2. + a)*muij(K)**2*(6. + a*muij(K)**2)) +
                      k2**5*(15. + a*muij(K)**2*
                             (45.+(a-2.)*muij(K)**2*(15.+(a-4.)*muij(K)**2)))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*k3**5))
        elif bc == '16':
            return ((-2.*pi*(-1. + (-1)**a)*
                     ((2. + a)*(4. + a)*(6. + a)*k1**6*muij(K) +
                      15.*(2. + a)*(4. + a)*k1**4*k2**2*muij(K)*
                      (3. + (3. + a)*muij(K)**2) +
                      6.*(2. + a)*(4. + a)*k1**5*k2*(1. + (5. + a)*muij(K)**2) +
                      15.*(2. + a)*k1**2*k2**4*muij(K)*
                      (15.+10.*(1. + a)*muij(K)**2 + (-1. + a**2)*muij(K)**4) +
                      20.*(2. + a)*k1**3*k2**3*
                      (3. + (3. + a)*muij(K)**2*(6. + (1. + a)*muij(K)**2)) +
                      k2**6*muij(K)*(105. + (-1. + a)*
                                     muij(K)**2*(105.+(a-3.)*muij(K)**2*
                                                 (21.+(a-5.)*muij(K)**2))) +
                      6*k1*k2**5*(15. + (1. + a)*muij(K)**2*
                                  (45.+(a-1.)*muij(K)**2*
                                   (15. + (-3. + a)*muij(K)**2)))))/
                    ((2 + a)*(4 + a)*(6 + a)*(8 + a)*k3**6))

        elif bc == '20':
            return ((2.*pi*(1.+(-1)**a)*(1.+a*muij(K)**2)) / ((1.+a)*(3.+a)))
        elif bc == '21':
            return ((2.*pi * (-1.+(-1)**a) *
                     (k1 + (1.+a)*k1*muij(K)**2 +
                      k2*muij(K)*(3.+(a-1.)*muij(K)**2))) /
                    ((2.+a)*(4.+a)*k3))
        elif bc == '22':
            return ((2.*pi * (1.+(-1)**a) *
                     (2.*(1.+a)*k1*k2*muij(K)*(3.+a*muij(K)**2)+
                      k1**2*(1.+a+(1.+a)*(2.+a)*muij(K)**2) +
                      k2**2*(3.+6.*a*muij(K)**2+(-2.+a)*a*
                             muij(K)**4))) / ((1.+a)*(3.+a)*(5.+a)*k3**2))
        elif bc == '23':
            return ((2.*pi * (-1.+(-1)**a) *
                     (3.*(2.+a)*k1**2*k2*muij(K)*
                      (3.+(1.+a)*muij(K)**2) +
                      k1**3*(2.+a+(2.+a)*(3.+a)*muij(K)**2) +
                      3.*k1*k2**2*(3.+6.*(1.+a)*muij(K)**2 +
                                   (a**2-1.)*muij(K)**4) +
                      k2**3*muij(K)*(15.+(a-1.)*muij(K)**2*
                                     (10.+(a-3.)*muij(K)**2)))) /
                    ((2.+a)*(4.+a)*(6.+a)*k3**3))
        elif bc == '24':
            return ((2.*pi*(1. + (-1)**a)*
                     (4*(1. + a)*(3. + a)*k1**3*k2*muij(K)*
                      (3. + (2. + a)*muij(K)**2) +
                      (1. + a)*(3. + a)*k1**4*(1. + (4. + a)*muij(K)**2) +
                      4.*(1. + a)*k1*k2**3*muij(K)*
                      (15. + 10*a*muij(K)**2 + (-2. + a)*a*muij(K)**4) +
                      6.*(1. + a)*k1**2*k2**2*
                      (3. + (2. + a)*muij(K)**2*(6. + a*muij(K)**2)) +
                      k2**4*(15. + a*muij(K)**2*
                             (45.+(a-2.)*muij(K)**2*(15.+(a-4.)*muij(K)**2)))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*k3**4))
        elif bc == '25':
            return ((2.*pi*(-1. + (-1)**a)*
                     (5.*(2. + a)*(4. + a)*k1**4*k2*muij(K)*
                      (3. + (3. + a)*muij(K)**2) +
                      (2. + a)*(4. + a)*k1**5*(1. + (5. + a)*muij(K)**2) +
                      10.*(2. + a)*k1**2*k2**3*muij(K)*
                      (15. + 10*(1. + a)*muij(K)**2 + (-1. + a**2)*muij(K)**4) +
                      10.*(2. + a)*k1**3*k2**2*
                      (3. + (3. + a)*muij(K)**2*(6. + (1. + a)*muij(K)**2)) +
                      k2**5*muij(K)*(105. + (a-1.)*muij(K)**2*
                                     (105.+(a-3.)*muij(K)**2*(21.+(a-5.)*muij(K)**2))) +
                      5.*k1*k2**4*(15.+(1.+a)*muij(K)**2*
                                   (45.+(a-1.)*muij(K)**2*(15.+(a-3.)*muij(K)**2)))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*k3**5))

        elif bc == '30':
            return -((2.*pi * (-1.+(-1)**a) *
                      muij(K)*(3.+(a-1.) * muij(K)**2))/((2.+a)*(4.+a)))
        elif bc == '31':
            return -((2.*pi * (1.+(-1)**a) *
                      ((1.+a)*k1*muij(K)*(3.+a*muij(K)**2) +
                       k2*(3.+6.*a*muij(K)**2 +
                           (a-2.)*a*muij(K)**4))) / ((1.+a)*(3.+a)*(5.+a)*k3))
        elif bc == '32':
            return -((2.*pi * (-1.+(-1)**a) *
                      ((2.+a)*k1**2*muij(K)*(3.+(1.+a)*muij(K)**2) +
                       2.*k1*k2*(3.+6.*(1.+a)*muij(K)**2+(a**2-1.)*
                                 muij(K)**4) + k2**2*muij(K)*
                       (15.+(a-1.)*muij(K)**2*
                        (10.+(a-3.)*muij(K)**2)))) /
                     ((2.+a)*(4.+a)*(6.+a)*k3**2))
        elif bc == '33':
            return ((-2.*pi*(1 + (-1)**a)*
                     ((1. + a)*(3. + a)*k1**3*muij(K)*
                      (3. + (2. + a)*muij(K)**2) + 3.*(1. + a)*k1*k2**2*muij(K)*
                      (15. + 10*a*muij(K)**2 + (-2. + a)*a*muij(K)**4) +
                      3.*(1.+a)*k1**2*k2*(3.+(2.+a)*muij(K)**2*
                                          (6. + a*muij(K)**2)) + k2**3*
                      (15.+a*muij(K)**2*(45.+(a-2.)*muij(K)**2*
                                         (15.+(a-4.)*muij(K)**2)))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*k3**3))
        elif bc == '34':
            return -((2.*pi * (-1.+(-1)**a) *
                      ((2.+a)*(4.+a)*k1**4*muij(K)*
                       (3.+(3.+a)*muij(K)**2) +
                       6.*(2.+a)*k1**2*k2**2*muij(K)*
                       (15.+10.*(1.+a)*muij(K)**2 + (a**2-1.)*
                        muij(K)**4) + 4.*(2.+a)*k1**3*k2*
                       (3.+(3.+a)*muij(K)**2* (6.+(1.+a)*muij(K)**2))+
                       k2**4*muij(K)*(105.+(a-1.)*muij(K)**2*
                                      (105.+(a-3.)*muij(K)**2*
                                       (21.+(a-5.)*muij(K)**2))) +
                       4.*k1*k2**3*(15.+(1.+a)*muij(K)**2*
                                    (45.+(a-1.)*muij(K)**2*
                                     (15.+(a-3.)*muij(K)**2))))) /
                     ((2.+a)*(4.+a)*(6.+a)*(8.+a)*k3**4))
        elif bc == '36':
            return ((-2.*pi*(-1. + (-1)**a)*
                     ((2. + a)*(4. + a)*(6. + a)*k1**6*muij(K)*
                      (3. + (5. + a)*muij(K)**2) +
                      15.*(2. + a)*(4. + a)*k1**4*k2**2*muij(K)*
                      (15. + (3. + a)*muij(K)**2*(10. + (1. + a)*muij(K)**2)) +
                      6.*(2. + a)*(4. + a)*k1**5*k2*
                      (3. + (5. + a)*muij(K)**2*(6. + (3. + a)*muij(K)**2)) +
                      20.*(2. + a)*k1**3*k2**3*
                      (15. + (3. + a)*muij(K)**2*
                       (45.+15.*(1.+a)*muij(K)**2 + (-1. + a**2)*muij(K)**4)) +
                      15.*(2. + a)*k1**2*k2**4*muij(K)*
                      (105. + (1. + a)*muij(K)**2*
                       (105.+(a-1.)*muij(K)**2*(21. + (-3. + a)*muij(K)**2))) +
                      k2**6*muij(K)*(945.+(a-1.)*muij(K)**2*
                                     (1260.+(a-3.)*muij(K)**2*
                                      (378.+(a-5.)*muij(K)**2*
                                       (36. + (-7. + a)*muij(K)**2)))) +
                      6.*k1*k2**5*(105.+(1.+a)*muij(K)**2*
                                   (420.+(a-1.)*muij(K)**2*
                                    (210.+(a-3.)*muij(K)**2*(28.+(a-5.)*muij(K)**2))))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*(10. + a)*k3**6))

        elif bc == '40':
            return ((2.*pi * (1.+(-1)**a) *
                     ((a-2.)*a*muij(K)**4 + 6*a* muij(K)**2 + 3.))/
                    ((1.+a)*(3.+a)*(5.+a)))
        elif bc == '41':
            return ((2.*pi * (-1.+(-1)**a) *
                     (k1*(3.+6*(1.+a)*muij(K)**2 + (-1.+a**2)* muij(K)**4) +
                      k2*muij(K)*(15.+(-1.+a)*muij(K)**2*
                                  (10.+(-3.+a)*muij(K)**2)))) /
                    ((2.+a)*(4.+a)*(6.+a)*k3))
        elif bc == '42':
            return ((2.*pi*(1 + (-1)**a)*
                     (2*(1. + a)*k1*k2*muij(K)*
                      (15. + 10*a*muij(K)**2 + (-2. + a)*a*muij(K)**4) +
                      (1. + a)*k1**2*(3 + (2. + a)*muij(K)**2*(6+a*muij(K)**2))+
                      k2**2*(15.+a*muij(K)**2*(45.+(a-2.)*muij(K)**2*
                                               (15.+(a-4.)*muij(K)**2)))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*k3**2))
        elif bc == '43':
            return ((2.*pi * (-1.+(-1)**a) *
                     (3.*(2.+a)*k1**2*k2*muij(K)*
                      (15.+10.*(1.+a)*muij(K)**2 +
                       (a**2-1.)*muij(K)**4) + (2.+a)*k1**3*
                      (3.+(3.+a)*muij(K)**2*
                       (6.+(1.+a)*muij(K)**2)) + k2**3*muij(K)*
                      (105.+(a-1.)*muij(K)**2*
                       (105.+(a-3.)*muij(K)**2*
                        (21.+(a-5)*muij(K)**2))) +
                      3.*k1*k2**2*(15.+(1.+a)*muij(K)**2*
                                   (45.+(a-1.)*muij(K)**2*
                                    (15.+(a-3.)*muij(K)**2))))) /
                    ((2.+a)*(4.+a)*(6.+a)*(8.+a)*k3**3))
        elif bc == '44':
            return  ((2.*pi*(1 + (-1)**a)*
                      (4*(1. + a)*(3. + a)*k1**3*k2*muij(K)*
                       (15. + (2. + a)*muij(K)**2*(10. + a*muij(K)**2)) +
                       (1.+a)*(3.+a)*k1**4*
                       (3.+(4.+a)*muij(K)**2* (6.+(2. + a)*muij(K)**2)) +
                       6.*(1. + a)*k1**2*k2**2* (15. + (2. + a)*muij(K)**2*
                                                 (45.+15.*a*muij(K)**2+(a-2.)*
                                                  a*muij(K)**4)) +
                       4*(1. + a)*k1*k2**3*muij(K)*
                       (105.+a*muij(K)**2*(105.+(a-2.)*muij(K)**2*
                                           (21. + (-4. + a)*muij(K)**2))) +
                       k2**4*(105.+a*muij(K)**2*
                              (420.+(a-2.)*muij(K)**2*
                               (210.+(a-4.)*muij(K)**2*
                                (28.+(a-6.)*muij(K)**2))))))/
                     ((1. + a)*(3. + a)*(5. + a)*(7. + a)*(9. + a)*k3**4))
        elif bc == '45':
            return ((2.*pi*(-1 + (-1)**a)*
                     (5*(2. + a)*(4. + a)*k1**4*k2*muij(K)*
                      (15. + (3. + a)*muij(K)**2*(10. + (1. + a)*muij(K)**2)) +
                      (2.+a)*(4.+a)*k1**5*(3.+(5.+a)*muij(K)**2*
                                           (6. + (3. + a)*muij(K)**2)) +
                      10.*(2.+a)*k1**3*k2**2*(15.+(3.+a)*muij(K)**2*
                                              (45.+15*(1.+a)*muij(K)**2 +
                                               (-1. + a**2)*muij(K)**4)) +
                      10.*(2.+a)*k1**2*k2**3*muij(K)*(105.+(1.+a)*muij(K)**2*
                       (105.+(a-1.)*muij(K)**2*(21. + (-3. + a)*muij(K)**2))) +
                      k2**5*muij(K)*(945.+(a-1.)*muij(K)**2*
                                     (1260.+(a-3.)*muij(K)**2*
                                      (378.+(a-5.)*muij(K)**2*
                                       (36.+(a-7.)*muij(K)**2))))+
                      5*k1*k2**4*(105. + (1. + a)*muij(K)**2*
                                  (420. + (-1. + a)*muij(K)**2*
                                   (210. + (-3. + a)*muij(K)**2*
                                    (28. + (-5. + a)*muij(K)**2))))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*(10. + a)*k3**5))

        elif bc == '50':
            return  ((-2.*pi*(-1. + (-1)**a)*
                      muij(K)*(15.+(a-1.)*muij(K)**2*(10.+(a-3.)*muij(K)**2)))/
                     ((2. + a)*(4. + a)*(6. + a)))
        elif bc == '51':
            return ((-2.*pi*(1. + (-1)**a)*
                     ((1.+a)*k1*muij(K)*(15.+10.*a*muij(K)**2 +
                                         (-2. + a)*a*muij(K)**4) +
                      k2*(15.+a*muij(K)**2*(45.+(a-2.)*muij(K)**2*
                                            (15.+(a-4.)*muij(K)**2)))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*k3))
        elif bc == '52':
            return ((-2.*pi*(-1 + (-1)**a)*
                     ((2. + a)*k1**2*muij(K)*(15. + 10*(1. + a)*muij(K)**2 +
                                              (-1. + a**2)*muij(K)**4) +
                      k2**2*muij(K)*(105. + (-1. + a)*muij(K)**2*
                                     (105.+(a-3.)*muij(K)**2*
                                      (21.+(a-5.)*muij(K)**2))) +
                      2*k1*k2*(15.+(1.+a)*muij(K)**2*
                               (45.+(a-1.)*muij(K)**2*
                                (15.+(a-3.)*muij(K)**2)))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*k3**2))
        elif bc == '53':
            return ((-2.*pi*(1. + (-1)**a)*
                     ((1. + a)*(3. + a)*k1**3*muij(K)*
                      (15. + (2. + a)*muij(K)**2*(10. + a*muij(K)**2)) +
                      3*(1. + a)*k1**2*k2*(15. + (2. + a)*muij(K)**2*
                                           (45.+15*a*muij(K)**2 +
                                            (a-2.)*a*muij(K)**4)) +
                      3*(1. + a)*k1*k2**2*muij(K)*
                      (105. + a*muij(K)**2*(105.+(a-2.)*muij(K)**2*
                                            (21. + (-4. + a)*muij(K)**2))) +
                      k2**3*(105.+a*muij(K)**2*(420.+(a-2.)*muij(K)**2*
                                                (210.+(a-4.)*muij(K)**2*
                                                 (28.+(a-6.)*muij(K)**2))))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*(9. + a)*k3**3))
        elif bc == '54':
            return ((-2.*pi*(-1. + (-1)**a)*
                     ((2. + a)*(4. + a)*k1**4*muij(K)*
                      (15. + (3. + a)*muij(K)**2*(10. + (1. + a)*muij(K)**2)) +
                      4.*(2. + a)*k1**3*k2*
                      (15. + (3. + a)*muij(K)**2*
                       (45.+15*(1.+a)*muij(K)**2 + (-1. + a**2)*muij(K)**4)) +
                      6.*(2.+a)*k1**2*k2**2*muij(K)*
                      (105.+(1 + a)*muij(K)**2*
                       (105.+(a-1.)*muij(K)**2*(21. + (-3. + a)*muij(K)**2))) +
                      k2**4*muij(K)*(945.+(a-1.)*muij(K)**2*
                                     (1260.+(a-3.)*muij(K)**2*
                                      (378.+(a-5.)*muij(K)**2*
                                       (36. + (-7. + a)*muij(K)**2))))+
                      4*k1*k2**3*(105.+(1.+a)*muij(K)**2*
                                  (420.+(a-1.)*muij(K)**2*
                                   (210.+(a-3.)*muij(K)**2*
                                    (28.+(a-5.)*muij(K)**2))))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*(10. + a)*k3**4))

        elif bc == '60':
            return ((2.*pi*(1. + (-1)**a)*
                     (15.+a*muij(K)**2*
                      (45.+(a-2.)*muij(K)**2*(15. + (-4. + a)*muij(K)**2))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)))
        elif bc == '61':
            return ((2.*pi*(-1 + (-1)**a)*
                     (k2*muij(K)*(105.+(a-1.)*muij(K)**2*
                                  (105.+(a-3.)*muij(K)**2*
                                   (21.+(a-5.)*muij(K)**2))) +
                      k1*(15.+(1.+a)*muij(K)**2*
                          (45.+(a-1.)*muij(K)**2*
                           (15.+(a-3.)*muij(K)**2)))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*k3))
        elif bc == '62':
            return ((2.*pi*(1. + (-1)**a)*
                     ((1 + a)*k1**2*(15.+(2.+a)*muij(K)**2*
                                     (45.+15*a*muij(K)**2 +
                                      (a-2.)*a*muij(K)**4)) +
                      2.*(1.+a)*k1*k2*muij(K)*
                      (105.+a*muij(K)**2*
                       (105.+(a-2.)*muij(K)**2*(21. + (-4. + a)*muij(K)**2))) +
                      k2**2*(105.+a*muij(K)**2*
                             (420.+(a-2)*muij(K)**2*(210.+(a-4.)*muij(K)**2*
                                                     (28.+(a-6.)*muij(K)**2))))))/
                    ((1. + a)*(3. + a)*(5. + a)*(7. + a)*(9. + a)*k3**2))
        elif bc == '63':
            return ((2.*pi*(-1. + (-1)**a)*
                     ((2. + a)*k1**3*(15.+(3.+a)*muij(K)**2*
                                      (45.+15.*(1.+a)*muij(K)**2 +
                                       (-1. + a**2)*muij(K)**4)) +
                      3.*(2. + a)*k1**2*k2*muij(K)*
                      (105. + (1. + a)*muij(K)**2*
                       (105.+(a-1.)*muij(K)**2*(21. + (-3. + a)*muij(K)**2))) +
                      k2**3*muij(K)*(945.+(a-1.)*muij(K)**2*
                                     (1260.+(a-3.)*muij(K)**2*
                                      (378.+(a-5.)*muij(K)**2*
                                       (36.+(a-7.)*muij(K)**2))))+ 3*k1*k2**2*
                      (105.+(1.+a)*muij(K)**2*(420.+(a-1)*muij(K)**2*
                                               (210.+(a-3.)*muij(K)**2*
                                                (28.+(a-5.)*muij(K)**2))))))/
                    ((2. + a)*(4. + a)*(6. + a)*(8. + a)*(10. + a)*k3**3))

    K = [k1, k2, k3]

    x = [(x, y) for x in [0,1,2,3,4] for y in ['00', '01', '02', '03', '04',
                                               '10', '11', '12', '13', '14',
                                               '20', '21', '22', '23',
                                               '30', '31', '32', '34',
                                               '40', '41', '43']]

    I = dict((x[1] + str(x[0]), Ia(x[1], a + x[0], K)) for x in x)
    Imu1 = dict((x[1] + str(x[0]), Ia(x[1], a + x[0]+2, K)) for x in x)
    Imu2 = dict((x[1] + str(x[0]), Ia(str(int(x[1][0])+2)+x[1][1],
                                      a + x[0], K)) for x in x)
    Imu3 = dict((x[1] + str(x[0]), Ia(x[1][0]+str(int(x[1][1])+2),
                                      a + x[0], K)) for x in x)
    tI = dict((x[1] + str(x[0]),
               ((x[0] + int(x[1][0]) + int(x[1][1]))*Ia(x[1], a + x[0], K) - \
                x[0]*Ia(x[1], a + x[0] + 2, K) - int(x[1][0])*\
                Ia(str(int(x[1][0])+2) + x[1][1], a + x[0], K) - \
                int(x[1][1])*Ia(x[1][0]+str(int(x[1][1])+2), a + x[0], K)))
              for x in x)

    I0 = [I['000']]
    I0mu = [Imu1['000'], Imu2['000'], Imu3['000']]

    I2 =   [I['002'], I['200'], I['020']]
    I2mu = [(Imu1['002'], Imu2['002'], Imu3['002']),
            (Imu1['200'], Imu2['200'], Imu3['200']),
            (Imu1['020'], Imu2['020'], Imu3['020'])]
    tI2 = [tI['002'], tI['200'], tI['020']]

    I4 = [I['004'], I['400'], I['040']]
    I4mu = [(Imu1['004'], Imu2['004'], Imu3['004']),
            (Imu1['400'], Imu2['400'], Imu3['400']),
            (Imu1['040'], Imu2['040'], Imu3['040'])]
    tI4 = [tI['004'], tI['400'], tI['040']]

    I11 = [I['011'], I['110'], I['101']]
    I11mu = [(Imu1['011'], Imu2['011'], Imu3['011']),
             (Imu1['110'], Imu2['110'], Imu3['110']),
             (Imu1['101'], Imu2['101'], Imu3['101'])]
    tI11 = [tI['011'], tI['110'], tI['101']]

    I13 = [I['013'], I['031'], I['103'], I['130'], I['301'], I['310']]
    I13mu = [(Imu1['013'], Imu2['013'], Imu3['013']),
             (Imu1['031'], Imu2['031'], Imu3['031']),
             (Imu1['103'], Imu2['103'], Imu3['103']),
             (Imu1['130'], Imu2['130'], Imu3['130']),
             (Imu1['301'], Imu2['301'], Imu3['301']),
             (Imu1['310'], Imu2['310'], Imu3['310'])]
    tI13 = [tI['013'], tI['031'], tI['103'], tI['130'], tI['301'], tI['310']]

    I112 = [I['112'], I['121'], I['211']]
    I112mu = [(Imu1['112'], Imu2['112'], Imu3['112']),
              (Imu1['121'], Imu2['121'], Imu3['121']),
              (Imu1['211'], Imu2['211'], Imu3['211'])]
    tI112 = [tI['112'], tI['121'], tI['211']]

    I22 = [I['202'], I['022'], I['220']]
    I22mu = [(Imu1['202'], Imu2['202'], Imu3['202']),
             (Imu1['022'], Imu2['022'], Imu3['022']),
             (Imu1['220'], Imu2['220'], Imu3['220'])]
    tI22 = [tI['202'], tI['022'], tI['220']]

    I114 = [I['411'], I['114'], I['141']]
    I114mu = [(Imu1['411'], Imu2['411'], Imu3['411']),
              (Imu1['114'], Imu2['114'], Imu3['114']),
              (Imu1['141'], Imu2['141'], Imu3['141'])]
    tI114 = [tI['411'], tI['114'], tI['141']]

    I123 = [I['123'], I['132'], I['213'], I['231'], I['312'], I['321']]
    I123mu = [(Imu1['123'], Imu2['123'], Imu3['123']),
              (Imu1['132'], Imu2['132'], Imu3['132']),
              (Imu1['213'], Imu2['213'], Imu3['213']),
              (Imu1['231'], Imu2['231'], Imu3['231']),
              (Imu1['312'], Imu2['312'], Imu3['312']),
              (Imu1['321'], Imu2['321'], Imu3['321'])]
    tI123 = [tI['123'], tI['132'], tI['213'], tI['231'], tI['312'], tI['321']]

    I222 = [I['222']]
    I222mu = [Imu1['222'], Imu2['222'], Imu3['222']]
    tI222 = [tI['222']]

    I134 = [I['134'], I['143'], I['314'], I['341'], I['413'], I['431']]
    I134mu = [(Imu1['134'], Imu2['134'], Imu3['134']),
              (Imu1['143'], Imu2['143'], Imu3['143']),
              (Imu1['314'], Imu2['314'], Imu3['314']),
              (Imu1['341'], Imu2['341'], Imu3['341']),
              (Imu1['413'], Imu2['413'], Imu3['413']),
              (Imu1['431'], Imu2['431'], Imu3['431'])]
    tI134 = [tI['134'], tI['143'], tI['314'], tI['341'], tI['413'], tI['431']]

    # Create dict for each a, then save in pbj as self.Is_B0, self.Is_B2,
    # self.Is_B4 or called directly in the main code
    dic = {'I0': I0, 'I0mu': I0mu, 'I2': I2, 'I2mu': I2mu, 'tI2': tI2, 'I4': I4,
           'I4mu': I4mu, 'tI4': tI4, 'I11': I11, 'I11mu': I11mu, 'tI11': tI11,
           'I13': I13, 'I13mu': I13mu, 'tI13': tI13, 'I112': I112,
           'I112mu': I112mu, 'tI112': tI112, 'I22': I22, 'I22mu': I22mu,
           'tI22': tI22, 'I114': I114, 'I114mu': I114mu, 'tI114': tI114,
           'I123': I123, 'I123mu': I123mu, 'tI123': tI123, 'I222': I222,
           'I222mu': I222mu, 'tI222': tI222,  'I134': I134, 'I134mu': I134mu,
           'tI134': tI134}
    return dic
