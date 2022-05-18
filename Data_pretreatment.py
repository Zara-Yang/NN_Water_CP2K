import os
import numpy as np
from tqdm import tqdm
from glob import glob
import subprocess

nAtom = 192
nO = 64
nH = 128
nW = 256

def Produce_split_features(data_path,script_path):
    """
    Copy and execute Produce_features.o in every config folder

    Input :
    Name                    description                               dimension
    --------------------------------------------------------------------------------
    data_path       folder path which conrtain config folder            None
    script_path     cpp executable file path                            None

    Output:
        None
    """
    config_path_list = list(i for i in glob("{}/*".format(data_path)))
    for config_path in tqdm(config_path_list):
        subprocess.call(["cp",script_path,config_path])
        subprocess.call(["./{}".format(os.path.basename(script_path))],cwd=config_path)

class Data_assemble():
    @staticmethod
    def Assemble_single_feature(config_folder_path,config_list,save_folder_path,feature_name, center_type):
        if center_type == "N" :
            ncenter = nNa
        elif center_type == "C":
            ncenter = nCl
        elif center_type == "n":
            ncenter = nNa
        elif center_type == "c":
            ncenter = nCl

        features_all = ()
        features_d_all = ()
        for config_folder_name in tqdm(config_list):
            feature_path = "{}/{}/features/feature_{}.txt".format(config_folder_path,config_folder_name,feature_name)
            feature_d_path = "{}/{}/features/dfeature_{}.txt".format(config_folder_path,config_folder_name,feature_name)
            features = np.loadtxt(feature_path, dtype=np.float32)
            if len(features.shape) == 1:
                features = np.expand_dims(features,axis=0)
            dfeatures = np.loadtxt(feature_d_path, dtype=np.float32)
            features_all += (features, )
            features_d_all += (dfeatures.reshape((features.shape[0], ncenter, 2 * nAtom, 3)), )

        features_all = np.transpose(np.stack(features_all, axis=0), axes=(0, 2, 1))  # now it is (nconfig, ncenter, nfeatures)
        features_d_all = np.transpose(np.stack(features_d_all, axis=0), axes=(0, 2, 3, 4, 1))  # now it is (nconfig, ncenter, natoms, 3, nfeatures)
        np.save("{}/features_{}".format(save_folder_path,feature_name), features_all)
        np.save("{}/features_d{}".format(save_folder_path,feature_name), features_d_all)
    @staticmethod
    def Assemble_all_features(features_name_tuple, features_folder_path):
        features_all = ()
        features_d_all = ()

        for feature_name in tqdm(features_name_tuple):
            features_d = np.load("{}/features_d{}".format(features_folder_path,feature_name) + ".npy")
            features = np.load("{}/features_{}".format(features_folder_path,feature_name) + ".npy")
            features_all += (features,)
            features_d_all += (features_d,)
        features_all = np.concatenate(features_all, axis=-1)  # stack along the nfeatures axis
        features_d_all = np.transpose(np.concatenate(features_d_all, axis=-1), axes=(0, 2, 3, 1, 4))
        # stack along the nfeatures axis, then make sure the number of center atoms is at the second last axis
        return features_all, features_d_all
    @staticmethod
    def Core_train_data(config_folder_path, config_list, features_list, save_folder_path,reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])

        xN,xN_d = Data_assemble.Assemble_all_features(["HS_Nn","EC_NN","EC_NC","EC_Nn","EC_Nc"], feature_path)
        xC,xC_d = Data_assemble.Assemble_all_features(["HS_Cc","EC_CN","EC_CC","EC_Cn","EC_Cc"], feature_path)

        xNN_d = xN_d[:,:nNa,:,:,:]
        xCN_d = xN_d[:,nNa:nAtom,:,:,:]
        xnN_d = xN_d[:,nAtom:nAtom + nNa,:,:,:]
        xcN_d = xN_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xNC_d = xC_d[:,:nNa,:,:,:]
        xCC_d = xC_d[:,nNa:nAtom,:,:,:]
        xnC_d = xC_d[:,nAtom:nAtom + nNa,:,:,:]
        xcC_d = xC_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xN_av = np.mean(xN, axis=(0, 1))
        xN_min = np.min(xN, axis=(0, 1))
        xN_max = np.max(xN, axis=(0, 1))

        xC_av = np.mean(xC, axis=(0, 1))
        xC_min = np.min(xC, axis=(0, 1))
        xC_max = np.max(xC, axis=(0, 1))
        np.savetxt("{}/xN_scalefactor.txt".format(save_folder_path), np.stack((xN_av, xN_min, xN_max), axis=-1))
        np.savetxt("{}/xC_scalefactor.txt".format(save_folder_path), np.stack((xC_av, xC_min, xC_max), axis=-1))

        xN = (xN - xN_av) / (xN_max - xN_min)
        xC = (xC - xC_av) / (xC_max - xC_min)

        xNN_d = xNN_d / (xN_max - xN_min)
        xCN_d = xCN_d / (xN_max - xN_min)
        xnN_d = xnN_d / (xN_max - xN_min)
        xcN_d = xcN_d / (xN_max - xN_min)

        xNC_d = xNC_d / (xC_max - xC_min)
        xCC_d = xCC_d / (xC_max - xC_min)
        xnC_d = xnC_d / (xC_max - xC_min)
        xcC_d = xcC_d / (xC_max - xC_min)

        print("Start saving npy : ")
        np.save("{}/xN".format(save_folder_path), xN)
        np.save("{}/xNNd".format(save_folder_path), xNN_d)
        np.save("{}/xCNd".format(save_folder_path), xCN_d)
        np.save("{}/xnNd".format(save_folder_path), xnN_d)
        np.save("{}/xcNd".format(save_folder_path), xcN_d)

        np.save("{}/xC".format(save_folder_path), xC)
        np.save("{}/xNCd".format(save_folder_path), xNC_d)
        np.save("{}/xCCd".format(save_folder_path), xCC_d)
        np.save("{}/xnCd".format(save_folder_path), xnC_d)
        np.save("{}/xcCd".format(save_folder_path), xcC_d)
    @staticmethod
    def Core_valid_data(config_folder_path, config_list, features_list, save_folder_path, rescale_file_path, reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])

        xN,xN_d = Data_assemble.Assemble_all_features(["HS_Nn","EC_NN","EC_NC","EC_Nn","EC_Nc"], feature_path)
        xC,xC_d = Data_assemble.Assemble_all_features(["HS_Cc","EC_CN","EC_CC","EC_Cn","EC_Cc"], feature_path)

        xNN_d = xN_d[:,:nNa,:,:,:]
        xCN_d = xN_d[:,nNa:nAtom,:,:,:]
        xnN_d = xN_d[:,nAtom:nAtom + nNa,:,:,:]
        xcN_d = xN_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xNC_d = xC_d[:,:nNa,:,:,:]
        xCC_d = xC_d[:,nNa:nAtom,:,:,:]
        xnC_d = xC_d[:,nAtom:nAtom + nNa,:,:,:]
        xcC_d = xC_d[:,nAtom + nNa : 2 * nAtom,: , :, :]
        xN_scalefector = np.loadtxt("{}/xN_scalefactor.txt".format(rescale_file_path))
        xC_scalefector = np.loadtxt("{}/xC_scalefactor.txt".format(rescale_file_path))

        xN_av  = xN_scalefector[:,0]
        xN_min = xN_scalefector[:,1]
        xN_max = xN_scalefector[:,2]

        xC_av  = xC_scalefector[:,0]
        xC_min = xC_scalefector[:,1]
        xC_max = xC_scalefector[:,2]
        xN = (xN - xN_av) / (xN_max - xN_min)
        xC = (xC - xC_av) / (xC_max - xC_min)

        xNN_d = xNN_d / (xN_max - xN_min)
        xCN_d = xCN_d / (xN_max - xN_min)
        xnN_d = xnN_d / (xN_max - xN_min)
        xcN_d = xcN_d / (xN_max - xN_min)

        xNC_d = xNC_d / (xC_max - xC_min)
        xCC_d = xCC_d / (xC_max - xC_min)
        xnC_d = xnC_d / (xC_max - xC_min)
        xcC_d = xcC_d / (xC_max - xC_min)
        print("Start saving npy : ")
        np.save("{}/xN".format(save_folder_path), xN)
        np.save("{}/xNNd".format(save_folder_path), xNN_d)
        np.save("{}/xCNd".format(save_folder_path), xCN_d)
        np.save("{}/xnNd".format(save_folder_path), xnN_d)
        np.save("{}/xcNd".format(save_folder_path), xcN_d)

        np.save("{}/xC".format(save_folder_path), xC)
        np.save("{}/xNCd".format(save_folder_path), xNC_d)
        np.save("{}/xCCd".format(save_folder_path), xCC_d)
        np.save("{}/xnCd".format(save_folder_path), xnC_d)
        np.save("{}/xcCd".format(save_folder_path), xcC_d)
    @staticmethod
    def Shell_train_data(config_folder_path, config_list, features_list, save_folder_path,reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])

        xn, xn_d = Data_assemble.Assemble_all_features(["HS_nN", "EC_nN", "EC_nC", "EC_nn", "EC_nc", "SH_nn", "SH_nc"],feature_path)
        xc, xc_d = Data_assemble.Assemble_all_features(["HS_cC", "EC_cN", "EC_cC", "EC_cn", "EC_cc", "SH_cn", "SH_cc"],feature_path)

        xNn_d = xn_d[:,:nNa,:,:,:]
        xCn_d = xn_d[:,nNa:nAtom,:,:,:]
        xnn_d = xn_d[:,nAtom:nAtom + nNa,:,:,:]
        xcn_d = xn_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xNc_d = xc_d[:,:nNa,:,:,:]
        xCc_d = xc_d[:,nNa:nAtom,:,:,:]
        xnc_d = xc_d[:,nAtom:nAtom + nNa,:,:,:]
        xcc_d = xc_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xn_av = np.mean(xn, axis=(0, 1))
        xn_min = np.min(xn, axis=(0, 1))
        xn_max = np.max(xn, axis=(0, 1))

        xc_av = np.mean(xc, axis=(0, 1))
        xc_min = np.min(xc, axis=(0, 1))
        xc_max = np.max(xc, axis=(0, 1))

        np.savetxt("{}/xn_scalefactor.txt".format(save_folder_path), np.stack((xn_av, xn_min, xn_max), axis=-1))
        np.savetxt("{}/xc_scalefactor.txt".format(save_folder_path), np.stack((xc_av, xc_min, xc_max), axis=-1))
        xn = (xn - xn_av) / (xn_max - xn_min)
        xc = (xc - xc_av) / (xc_max - xc_min)

        xNn_d = xNn_d / (xn_max - xn_min)
        xCn_d = xCn_d / (xn_max - xn_min)
        xnn_d = xnn_d / (xn_max - xn_min)
        xcn_d = xcn_d / (xn_max - xn_min)

        xNc_d = xNc_d / (xc_max - xc_min)
        xCc_d = xCc_d / (xc_max - xc_min)
        xnc_d = xnc_d / (xc_max - xc_min)
        xcc_d = xcc_d / (xc_max - xc_min)
        print("Start saving npy : ")
        np.save("{}/xn".format(save_folder_path), xn)
        np.save("{}/xNnd".format(save_folder_path), xNn_d)
        np.save("{}/xCnd".format(save_folder_path), xCn_d)
        np.save("{}/xnnd".format(save_folder_path), xnn_d)
        np.save("{}/xcnd".format(save_folder_path), xcn_d)

        np.save("{}/xc".format(save_folder_path), xc)
        np.save("{}/xNcd".format(save_folder_path), xNc_d)
        np.save("{}/xCcd".format(save_folder_path), xCc_d)
        np.save("{}/xncd".format(save_folder_path), xnc_d)
        np.save("{}/xccd".format(save_folder_path), xcc_d)
    @staticmethod
    def Shell_valid_data(config_folder_path, config_list, features_list, save_folder_path, rescale_file_path, reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])

        xn, xn_d = Data_assemble.Assemble_all_features(["HS_nN", "EC_nN", "EC_nC", "EC_nn", "EC_nc", "SH_nn", "SH_nc"],feature_path)
        xc, xc_d = Data_assemble.Assemble_all_features(["HS_cC", "EC_cN", "EC_cC", "EC_cn", "EC_cc", "SH_cn", "SH_cc"],feature_path)

        xNn_d = xn_d[:,:nNa,:,:,:]
        xCn_d = xn_d[:,nNa:nAtom,:,:,:]
        xnn_d = xn_d[:,nAtom:nAtom + nNa,:,:,:]
        xcn_d = xn_d[:,nAtom + nNa : 2 * nAtom,: , :, :]

        xNc_d = xc_d[:,:nNa,:,:,:]
        xCc_d = xc_d[:,nNa:nAtom,:,:,:]
        xnc_d = xc_d[:,nAtom:nAtom + nNa,:,:,:]
        xcc_d = xc_d[:,nAtom + nNa : 2 * nAtom,: , :, :]
        xn_scalefector = np.loadtxt("{}/xn_scalefactor.txt".format(rescale_file_path))
        xc_scalefector = np.loadtxt("{}/xc_scalefactor.txt".format(rescale_file_path))

        xn_av  = xn_scalefector[:,0]
        xn_min = xn_scalefector[:,1]
        xn_max = xn_scalefector[:,2]

        xc_av  = xc_scalefector[:,0]
        xc_min = xc_scalefector[:,1]
        xc_max = xc_scalefector[:,2]
        xn = (xn - xn_av) / (xn_max - xn_min)
        xc = (xc - xc_av) / (xc_max - xc_min)

        xNn_d = xNn_d / (xn_max - xn_min)
        xCn_d = xCn_d / (xn_max - xn_min)
        xnn_d = xnn_d / (xn_max - xn_min)
        xcn_d = xcn_d / (xn_max - xn_min)

        xNc_d = xNc_d / (xc_max - xc_min)
        xCc_d = xCc_d / (xc_max - xc_min)
        xnc_d = xnc_d / (xc_max - xc_min)
        xcc_d = xcc_d / (xc_max - xc_min)
        print("Start saving npy : ")
        np.save("{}/xn".format(save_folder_path), xn)
        np.save("{}/xNnd".format(save_folder_path), xNn_d)
        np.save("{}/xCnd".format(save_folder_path), xCn_d)
        np.save("{}/xnnd".format(save_folder_path), xnn_d)
        np.save("{}/xcnd".format(save_folder_path), xcn_d)

        np.save("{}/xc".format(save_folder_path), xc)
        np.save("{}/xNcd".format(save_folder_path), xNc_d)
        np.save("{}/xCcd".format(save_folder_path), xCc_d)
        np.save("{}/xncd".format(save_folder_path), xnc_d)
        np.save("{}/xccd".format(save_folder_path), xcc_d)
    @staticmethod
    def Core_only_train_data(config_folder_path, config_list, features_list, save_folder_path,reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])
        xN,xN_d = Data_assemble.Assemble_all_features(["EC_NN","EC_NC","SH_NN","SH_NC"], feature_path)
        xC,xC_d = Data_assemble.Assemble_all_features(["EC_CN","EC_CC","SH_CN","SH_CC"], feature_path)

        xNN_d = xN_d[:,:nNa,:,:,:]
        xCN_d = xN_d[:,nNa:nAtom,:,:,:]
        xNC_d = xC_d[:,:nNa,:,:,:]
        xCC_d = xC_d[:,nNa:nAtom,:,:,:]

        xN_av = np.mean(xN, axis=(0, 1))
        xN_min = np.min(xN, axis=(0, 1))
        xN_max = np.max(xN, axis=(0, 1))

        xC_av = np.mean(xC, axis=(0, 1))
        xC_min = np.min(xC, axis=(0, 1))
        xC_max = np.max(xC, axis=(0, 1))
        np.savetxt("{}/xN_scalefactor.txt".format(save_folder_path), np.stack((xN_av, xN_min, xN_max), axis=-1))
        np.savetxt("{}/xC_scalefactor.txt".format(save_folder_path), np.stack((xC_av, xC_min, xC_max), axis=-1))

        xN = (xN - xN_av) / (xN_max - xN_min)
        xC = (xC - xC_av) / (xC_max - xC_min)

        xNN_d = xNN_d / (xN_max - xN_min)
        xCN_d = xCN_d / (xN_max - xN_min)
        xNC_d = xNC_d / (xC_max - xC_min)
        xCC_d = xCC_d / (xC_max - xC_min)
        print("Start saving npy : ")

        np.save("{}/xN".format(save_folder_path), xN)
        np.save("{}/xNNd".format(save_folder_path), xNN_d)
        np.save("{}/xCNd".format(save_folder_path), xCN_d)

        np.save("{}/xC".format(save_folder_path), xC)
        np.save("{}/xNCd".format(save_folder_path), xNC_d)
        np.save("{}/xCCd".format(save_folder_path), xCC_d)
    @staticmethod
    def Core_only_valid_data(config_folder_path, config_list, features_list, save_folder_path, rescale_file_path, reassemble = False):
        feature_path = "{}/features".format(save_folder_path)
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
        if reassemble:
            for feature_name in features_list:
                Data_assemble.Assemble_single_feature(config_folder_path, config_list, feature_path, feature_name, feature_name[3])

        xN,xN_d = Data_assemble.Assemble_all_features(["EC_NN","EC_NC","SH_NN","SH_NC"], feature_path)
        xC,xC_d = Data_assemble.Assemble_all_features(["EC_CN","EC_CC","SH_CN","SH_CC"], feature_path)

        xNN_d = xN_d[:,:nNa,:,:,:]
        xCN_d = xN_d[:,nNa:nAtom,:,:,:]
        xNC_d = xC_d[:,:nNa,:,:,:]
        xCC_d = xC_d[:,nNa:nAtom,:,:,:]
        xN_scalefector = np.loadtxt("{}/xN_scalefactor.txt".format(rescale_file_path))
        xC_scalefector = np.loadtxt("{}/xC_scalefactor.txt".format(rescale_file_path))

        xN_av  = xN_scalefector[:,0]
        xN_min = xN_scalefector[:,1]
        xN_max = xN_scalefector[:,2]

        xC_av  = xC_scalefector[:,0]
        xC_min = xC_scalefector[:,1]
        xC_max = xC_scalefector[:,2]
        xN = (xN - xN_av) / (xN_max - xN_min)
        xC = (xC - xC_av) / (xC_max - xC_min)

        xNN_d = xNN_d / (xN_max - xN_min)
        xCN_d = xCN_d / (xN_max - xN_min)
        xNC_d = xNC_d / (xC_max - xC_min)
        xCC_d = xCC_d / (xC_max - xC_min)
        print("Start saving npy : ")
        np.save("{}/xN".format(save_folder_path), xN)
        np.save("{}/xNNd".format(save_folder_path), xNN_d)
        np.save("{}/xCNd".format(save_folder_path), xCN_d)
        np.save("{}/xC".format(save_folder_path), xC)
        np.save("{}/xNCd".format(save_folder_path), xNC_d)
        np.save("{}/xCCd".format(save_folder_path), xCC_d)

if __name__ == "__main__":
    train_config_name = [i for i in range(5000)]
    valid_config_name = [i for i in range(5000,5829)]

    script_name = "Produce_features.o"
    config_folder_path = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/Split_data"

    TRAIN_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/TrainInput"
    VALID_INPUT_PATH = "/DATA/users/yanghe/projects/NeuralNetwork_PES/Neural_network/Water_BPNN/data/ValidInput"
    Produce_split_features(config_folder_path,script_name)

