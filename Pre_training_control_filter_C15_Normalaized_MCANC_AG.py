import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio

from Norimalized_Multichannel_FxLMS_algorithm import McFxNLMS_algorithm, train_McFxNLMS_algorithm, Disturbance_Noise_generation_from_Fvector
from scipy.io import savemat
from scipy import signal

#-----------------------------------------------------------------------------------
# Function: save_mat__()
# Description: Save the data into mat file for 
# Matlab.
#-----------------------------------------------------------------------------------
def save_mat__(FILE_NAME_PATH, Wc):
    mdict= {'Wc_v': Wc}
    savemat(FILE_NAME_PATH, mdict)

#-----------------------------------------------------------------------------------
# Function loading_paths_from_MAT（）
#-----------------------------------------------------------------------------------
def loading_paths_from_MAT(folder = 'Duct_path'
                           ,Pri_path_file_name = 'Primary_path.mat'
                           ,Sec_path_file_name ='Secondary_path.mat'):
    Primay_path_file, Secondary_path_file = os.path.join(folder, Pri_path_file_name), os.path.join(folder,Sec_path_file_name)
    Pri_dfs, Secon_dfs = sio.loadmat(Primay_path_file), sio.loadmat(Secondary_path_file)
    Pri_path, Secon_path = Pri_dfs['Pri_path'].squeeze(), Secon_dfs['Sec_path'].squeeze()
    return Pri_path, Secon_path

#-----------------------------------------------------------------------------------
# Class       : frequencyband_design()
# Description : The function is utilized to devide the full frequency band into 
# several equal frequency components.
#-----------------------------------------------------------------------------------
def frequencyband_design(level,fs):
    # the number of filter equals 2^level.
    # fs represents the sampling rate. 
    Num = 2**level
    # Computing the start and end of the frequency band.
    #----------------------------------------------------
    F_vector = []
    f_start  = 20
    f_marge  = 20 
    # the wideth of thefrequency band
    width    = (fs/2-f_start-f_marge)//Num 
    #----------------------------------------------------
    for ii in range(Num):
        f_end   = f_start + width 
        F_vector.append([f_start,f_end])
        f_start = f_end 
    #----------------------------------------------------
    return F_vector, width

#-----------------------------------------------------------------------------------
def main():
    FILE_NAME_PATH = 'Control_filter_from_15frequencies.mat'
    # Configurating the system parameters
    fs = 16000 
    T  = 30 
    Len_control = 1024 
    level          = 4 #4 
    
    Frequecy_band = []
    for i in range(level):
        F_vec, _       = frequencyband_design(i, fs)
        Frequecy_band += F_vec
    
    # Loading the primary and secondary path
    # Pri_path, Secon_path = loading_paths() 
    Pri_path, Secon_path = loading_paths_from_MAT()
    Sec = torch.from_numpy(Secon_path).type(torch.float).unsqueeze(0)
    
    # Training the control filters from the defined frequency band 
    num_filters = len(Frequecy_band)
    Wc_matrix   = np.zeros((num_filters, Len_control), dtype=float)
    
    print(Frequecy_band)
    
    
    for ii, F_vector in enumerate( Frequecy_band):
        print(F_vector)
        Dis, Re = Disturbance_Noise_generation_from_Fvector(fs=fs, T= T, f_vector=F_vector, Pri_path=Pri_path, Sec_path=Secon_path)
        #controller = FxLMS_AG_algroithm(Len=Len_control,)
        #----------------------------
        Re2   = torch.cat((Re.unsqueeze(0),Re.unsqueeze(0)),0)
        Dis2  = torch.cat((Dis.unsqueeze(0),Dis.unsqueeze(0)),0)
        SecM2 = torch.zeros(2,2,len(Sec))
        SecM2 = SecM2 + Sec.unsqueeze(0)
        print(Re2.shape)
        print(SecM2)
        #--------------------------------------------
        
        #--------------------------------------------
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        
        controller = McFxNLMS_algorithm(R_num=2, S_num=2, Len=Len_control, Sec=SecM2, device=device)
        
        Erro = train_McFxNLMS_algorithm(Model=controller, Ref=Re2, Disturbance=Dis2, Stepsize=0.01)
        Wc_matrix[ii] = controller._get_coeff_()[0,0,:]
        
        # Drawing the impulse response of the primary path
        plt.title('The error signal of the FxLMS algorithm')
        plt.plot(Erro)
        plt.ylabel('Amplitude')
        plt.xlabel('Time')
        plt.grid()
        plt.show()
        
        fs=16000
        f, Pper_spec = signal.periodogram(Wc_matrix[ii] , fs, 'flattop', scaling='spectrum')
        plt.semilogy(f, Pper_spec)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD')
        plt.grid()
        plt.show()
        
    save_mat__(FILE_NAME_PATH, Wc_matrix)

if __name__ == "__main__":
    main()