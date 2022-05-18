import os
import math
import numpy as np
from glob import glob
from tqdm import tqdm
from Ewald_sum import Ewald_sum

"""
    Prepare Input data and label for force training
"""

charge_O = +6
charge_H = +1
charge_W = -2
sigma = 8


class Data_Collect():
    @staticmethod
    def Collect_single_config(folder_path):
        H_coord = np.loadtxt("{}/Hxyz.txt".format(folder_path))
        O_coord = np.loadtxt("{}/Oxyz.txt".format(folder_path))
        W_coord = np.loadtxt("{}/Wxyz.txt".format(folder_path))

        O_force = np.loadtxt("{}/Oforce.txt".format(folder_path))
        H_force = np.loadtxt("{}/Hforce.txt".format(folder_path))

        Box = np.loadtxt("{}/Box.txt".format(folder_path))

        return(O_coord, H_coord, W_coord, O_force, H_force, Box)
    @staticmethod
    def Collect_all_config(folder_path):
        Hxyz = ()
        Oxyz = ()
        Wxyz = ()
        Hforce = ()
        Oforce = ()
        Box = ()
        for config_path in glob("{}/*".format(folder_path)):
            O_coord, H_coord, W_coord, O_force, H_force, box = Data_Collect.Collect_single_config(config_path)
            Hxyz += (H_coord,)
            Oxyz += (O_coord,)
            Wxyz += (W_coord,)
            Hforce += (H_force,)
            Oforce += (O_force,)
            Box += (box,)
        Hxyz = np.array(Hxyz)
        Oxyz = np.array(Oxyz)
        Wxyz = np.array(Wxyz)
        Hforce = np.array(Hforce)
        Oforce = np.array(Oforce)
        Box = np.array(Box)
        return(Oxyz, Hxyz, Wxyz, Oforce, Hforce, Box)
    @staticmethod
    def Get_Ewald(Oxyz, Hxyz, Wxyz, Box, save_path):
        Oxyz_extend = np.transpose(np.expand_dims(Oxyz, axis = 0), axes = (2, 3, 0, 1))
        Hxyz_extend = np.transpose(np.expand_dims(Hxyz, axis = 0), axes = (2, 3, 0, 1))
        Wxyz_extend = np.transpose(np.expand_dims(Wxyz, axis = 0), axes = (2, 3, 0, 1))
        Box_extend = np.transpose(Box, axes = (1, 0))

        EO, EH, EW = Ewald_sum(Oxyz_extend, Hxyz_extend, Wxyz_extend, Box_extend, charge_H, charge_O, charge_W, sigma, nkmax = 5)
        np.save( "{}/EO".format(save_path), EO)
        np.save( "{}/EH".format(save_path), EH)
        np.save( "{}/EW".format(save_path), EW)
    @staticmethod
    def Load_Ewald(EO_path, EH_path, EW_path):
        EO = np.load(EO_path)
        EH = np.load(EH_path)
        EW = np.load(EW_path)
        return(EO, EH, EW)
    @staticmethod
    def Calculate_long_force(EO_path, EH_path, EW_path, save_path):
        EO_extend, EH_extend, EW_extend = Data_Collect.Load_Ewald(EO_path, EH_path, EW_path)

        EO = np.squeeze(EO_extend, axis = 2)
        EH = np.squeeze(EH_extend, axis = 2)
        EW = np.squeeze(EW_extend, axis = 2)

        fO_long = EO * charge_O
        fH_long = EH * charge_H
        fW_long = EW * charge_W

        np.save("{}/fO_long".format(save_path), fO_long)
        np.save("{}/fH_long".format(save_path), fH_long)
        np.save("{}/fW_long".format(save_path), fW_long)

if __name__ == "__main__":
    Config_folder_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/Split_data/"
    Oxyz, Hxyz, Wxyz, Oforce, Hforce, Box = Data_Collect.Collect_all_config(Config_folder_path)
    Data_Collect.Calculate_long_force("./EO.npy", "./EH.npy", "./EW.npy", "./")









