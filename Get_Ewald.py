import os
import math
import numpy as np
from glob import glob
from tqdm import tqdm
"""
    Calculate Long range force for :
        a. Water with wannier center
        b. NaCl shell model

    Already check result with 108 NaCl shell model include
    Waring : In NaCl shell model, there is no electric force between core and own shell !

                                            ZaraYang
                                            2022 - 05 - 18
"""
# ===============  [ CONSTANT DEFINE ]  =====================
pi = math.pi
sqr_pi = math.sqrt(math.pi)
alpha = 1 / 4.5
r4pie = 138935.45764438206

chargeO = + 6
chargeH = + 1
chargeW = - 2

nAtom = 192
nO = 64
nH = 128
nW = 4 * nO
nkmax = 4

# ==============  [ START CALCULATE ]  ====================
class Get_Ewald():
    @staticmethod
    def Calculate_rij(coords, Box):
        """
            Calculate rij matrix with PBC
        """
        xyz_expand_1 = np.expand_dims(coords, axis = 0)
        xyz_expand_2 = np.expand_dims(coords, axis = 1)
        box_expand   = np.expand_dims(Box, axis = (0,1))
        vec_rij = xyz_expand_1 - xyz_expand_2
        vec_rij = vec_rij - np.round(vec_rij / box_expand) * box_expand
        # Plug only for NaCl shell model
        # ========================================
        # nNa = int(coords.shape[0]/4)
        # nAtom = 2 * nNa
        # for i in range(nAtom):
        #     vec_rij[i, i + nAtom] = np.zeros(3)
        #     vec_rij[i + nAtom, i] = np.zeros(3)
        # ========================================
        return(vec_rij)
    @staticmethod
    def Ewald_sum(coords, box, charges):
        """
        Calculate long range force

        1. Math :
                           4 * pi                k_vec              k^2
            Fi = - q_i * ----------  Sum_{k!=0} ------- exp( - -------------) * Sum_{j}^N q_j Sin(k_vec * r_ij_vec)
                             V                    k^2           4 * alpha^2
        2. Input variable :
            ===========================================================================
            Input            Shape                      unit         Indroduction
            ---------------------------------------------------------------------------
            coords          [nAtom + nWannier, 3]       ans         Total Particle coord
            box             [3]                         ans         Total box length
            charges         [nAtom + nWannier]           e          Particle charges
            ===========================================================================

        3. Outpur variable :
            ===========================================================================
            Output          Shape                   unit            Indroduction
            ---------------------------------------------------------------------------
            Force_long      [nAtom + nWannier, 3]   10J/(mol*Ans)   Long range force
            ===========================================================================

        4. Reference :
            a. Note from Yihao : Ewald force.pdf
        """
        k_unit = 2 * pi / box
        k_coord = np.zeros(((2 * nkmax + 1)**3-1, 3), dtype=np.float32)
        ik = 0
        for nx in range(- nkmax, nkmax + 1):
           for ny in range(- nkmax, nkmax + 1):
                for nz in range(- nkmax, nkmax + 1):
                    if ((nx != 0) | (ny != 0) | (nz != 0)):
                        k_coord[ik, 0] = nx * k_unit[0]
                        k_coord[ik, 1] = ny * k_unit[1]
                        k_coord[ik, 2] = nz * k_unit[2]
                        ik = ik + 1
        rij_vec = Get_Ewald.Calculate_rij(coords, box)
        kxyz_expand = np.expand_dims(k_coord, axis = (0, 1))
        q_expand = np.expand_dims(charges, axis = (0, 2))
        rij_expand = np.expand_dims(rij_vec, axis = 2 )
        Sk_buffer = np.sum(q_expand * np.sin(np.sum(rij_expand * kxyz_expand, axis = -1, keepdims = False)), axis = 1)  #[nAtom, nk]
        k_norm = np.linalg.norm(k_coord, axis = 1, keepdims = True)
        k_param = k_coord / np.power(k_norm, 2) * np.exp(- np.power(k_norm, 2)/(4 * alpha * alpha))
        Force = np.sum(np.expand_dims(k_param, axis = 0) * np.expand_dims(Sk_buffer,axis = 2), axis = 1)
        Force = Force * np.expand_dims(charges, axis = (1)) * 4 * pi / (box[0] * box[1] * box[2])
        Force = Force * r4pie * (-1)
        return(Force)
    @staticmethod
    def Run_Ewald(config_folder_path):
        charges = np.concatenate((np.ones(nO) * chargeO, np.ones(nH) * chargeH, np.ones(nW) * chargeW), axis = 0)
        for config_path in tqdm(glob("{}/*".format(config_folder_path))):
            print(config_path)
            box = np.loadtxt("{}/Box.txt".format(config_path))
            Oxyz = np.loadtxt("{}/Oxyz.txt".format(config_path))
            Hxyz = np.loadtxt("{}/Hxyz.txt".format(config_path))
            Wxyz = np.loadtxt("{}/Wxyz.txt".format(config_path))
            coords = np.concatenate((Oxyz, Hxyz, Wxyz), axis=0)
            force_long = Get_Ewald.Ewald_sum(coords, box, charges)
            np.savetxt("{}/Oforce_l.txt".format(config_path), force_long[:nO,:])
            np.savetxt("{}/Hforce_l.txt".format(config_path), force_long[nO:nAtom,:])
            np.savetxt("{}/Wforce_l.txt".format(config_path), force_long[nAtom:,:])
    @staticmethod
    def Run_Ewald_NaCl(config_path):
        """
            Calculate long range force for NaCl shell model system
            Waring : Remeber cancel  elect force between core and own shell !
        """
        ChargeNaShell = -0.50560
        ChargeClShell = -2.50050
        ChargeNaCore  = +1 - ChargeNaShell
        ChargeClCore  = -1 - ChargeClShell
        charges = np.concatenate((np.ones(108) * ChargeNaCore, np.ones(108) * ChargeClCore, np.ones(108) * ChargeNaShell, np.ones(108) * ChargeClShell), axis = 0)
        for config_path in glob("{}/*".format(config_folder_path)):
            print(config_path)
            box = np.loadtxt("{}/Box.txt".format(config_path))
            Nxyz = np.loadtxt("{}/R_Na.txt".format(config_path))
            Cxyz = np.loadtxt("{}/R_Cl.txt".format(config_path))
            nxyz = np.loadtxt("{}/r_Na.txt".format(config_path))
            cxyz = np.loadtxt("{}/r_Cl.txt".format(config_path))
            coords = np.concatenate((Nxyz, Cxyz, nxyz, cxyz), axis=0)
            force_long = Get_Ewald.Ewald_sum(coords, box, charges)
            print(force_long)
            exit()

if __name__ == "__main__":
    config_folder_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/Split_data/"
    Get_Ewald.Run_Ewald(config_folder_path)



