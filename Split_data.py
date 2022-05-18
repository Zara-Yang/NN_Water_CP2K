import os
import math
import numpy as np
from glob import glob
from tqdm import tqdm

"""
    Loading Yihao's CP2K H2O dataset

"""
class Split_data():
    @staticmethod
    def Split_matrix(Data, wannier):
        nParticle = Data.shape[0]
        nO = int(nParticle / 7) if wannier else int(nParticle / 3)
        nH = 2 * nO
        nW = 4 * nO
        Oxyz = np.zeros((nO, 3))
        Hxyz = np.zeros((nH, 3))
        Wxyz = np.zeros((nW, 3))

        for iconfig in range(nO):
            Oxyz[iconfig        ] = Data[3 * iconfig    ]
            Hxyz[iconfig * 2    ] = Data[3 * iconfig + 1]
            Hxyz[iconfig * 2 + 1] = Data[3 * iconfig + 2]
        if wannier :
            # Continue read wannier center
            for iwannier in range(nW):
                Wxyz[iwannier] = Data[nO + nH + iwannier]
        return(Oxyz, Hxyz, Wxyz)
    @staticmethod
    def Save_config(Oxyz, Hxyz, Wxyz, Box, Oforce, Hforce, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs("{}/features".format(save_path))
        np.savetxt("{}/Oxyz.txt".format(save_path), Oxyz)
        np.savetxt("{}/Hxyz.txt".format(save_path), Hxyz)
        np.savetxt("{}/Wxyz.txt".format(save_path), Wxyz)
        np.savetxt("{}/Box.txt".format(save_path),  Box)
        np.savetxt("{}/Oforce.txt".format(save_path), Oforce)
        np.savetxt("{}/Hforce.txt".format(save_path), Hforce)
    @staticmethod
    def Load_single_file(path):
        """
        ================================================================
        Loading coord or force in single file (single config or multconfig )

        ----------------------------------------------------------------

        Args :
            path(string) : path of loading file

        Return :
            Data(np.array) : coord or force in file

            Array shape : [ 1        nParticle, 3 ] ----> single config in file
                          [ nconfig, nParticle, 3 ] ----> multiple config in file

        ================================================================
        """
        Data = []
        fp = open(path,"r")
        for line in fp:
            line = line.strip().split()
            if len(line) == 1:
                nParticle = int(line[0])
                Data.append([])
                fp.readline()
                continue
            Data[-1].append([float(i) for i in line[1:]])
        Data = np.array(Data)
        return(Data)
    @staticmethod
    def Assemble_data(coord_path, force_path, Box, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        coords = Split_data.Load_single_file(coord_path)
        forces = Split_data.Load_single_file(force_path)
        print(coords.shape)
        print(forces.shape)
        nconfig, nparticle, _ = coords.shape
        for iconfig in tqdm(range(nconfig)):
            Oxyz, Hxyz, Wxyz = Split_data.Split_matrix(coords[iconfig], True)
            Oforce, Hforce, _= Split_data.Split_matrix(forces[iconfig], False)
            config_save_path = "{}/{}".format(save_path, iconfig)
            Split_data.Save_config(Oxyz, Hxyz, Wxyz, Box, Oforce, Hforce, config_save_path)
if __name__ == "__main__":
    Coord_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Public_data/Bulk64_Yihao/CP2KH2O/wannier.dat"
    Force_path =  "/DATA/users/yanghe/projects/NeuralNetwork_PES/Public_data/Bulk64_Yihao/CP2KH2O/force.dat"
    Save_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/Split_data/"
    Box = np.array([12.43142074624,12.43142074624,12.43142074624])
    Split_data.Assemble_data(Coord_path, Force_path, Box, Save_path)





