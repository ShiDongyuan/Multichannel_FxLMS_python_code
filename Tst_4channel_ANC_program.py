import os 
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio

from Norimalized_Multichannel_FxLMS_algorithm import McFxNLMS_algorithm, train_McFxNLMS_algorithm, Disturbance_Noise_generation_from_Fvector
from scipy.io import savemat
from scipy import signal

#------------------------------------------------------------
# Function : Save_data_to_mat()
#------------------------------------------------------------
def Save_data_to_mat(Mat_file_name ='Tst_4channel_program.mat', **kwargs):
     mdict = {}
     for arg in kwargs:
          mdict[arg] = kwargs[arg]
     savemat(Mat_file_name, mdict)

#------------------------------------------------------------
# Function : Load_noise_path_from_mat()
# Loading primary path and noise from mat.
#------------------------------------------------------------
def Load_noise_path_from_mat(Noise_mat_file='Noise_generation.mat', Path_mat_file='Path_generation.mat'):
     Refer_matrix  = torch.from_numpy(sio.loadmat(Noise_mat_file)['Refer_matrix']).type(torch.float)
     Distur_matrix = torch.from_numpy(sio.loadmat(Noise_mat_file)['Distur_matrix']).type(torch.float)
     Primary_path_matrix = torch.from_numpy(sio.loadmat(Path_mat_file)['Primary_path_matrix']).type(torch.float)
     Secondary_path_matrix = torch.from_numpy(sio.loadmat(Path_mat_file)['Secondary_path_matrix']).type(torch.float)
     
     return Primary_path_matrix, Secondary_path_matrix, Refer_matrix, Distur_matrix

#------------------------------------------------------------
if __name__ == "__main__":
     Pri_mat, Sec_mat, Refer_mat, Distur_mat = Load_noise_path_from_mat()
    
     Sec_path = torch.permute(Sec_mat,(2,0,1)) 
     Refer    = torch.permute(Refer_mat,(1,0))
     Distu    = torch.permute(Distur_mat,(1,0))
     print(Refer.shape)
    
    #--------------------------------------------
     Len_control = 512 
     num_filters = 16
     #Wc_matrix   = np.zeros((num_filters, Len_control), dtype=float)
     if torch.cuda.is_available():
          device = "cuda"
     else:
          device = "cpu"
        
     controller = McFxNLMS_algorithm(R_num=4, S_num=4, Len=Len_control, Sec=Sec_path, device=device)
     Erro       = train_McFxNLMS_algorithm(Model=controller, Ref=Refer, Disturbance=Distu, Stepsize=0.01)
     Wc_matrix  = controller._get_coeff_()
     
     # Saving the mat 
     Err_array = np.array(Erro)
     Save_data_to_mat(Mat_file_name ='Tst_4channel_program.mat', Wc_matrix=Wc_matrix, Err_array=Err_array)
     print(Err_array.shape)
        
     # Drawing the impulse response of the primary path
     plt.title('The error signal of the FxLMS algorithm')
     plt.plot(Err_array[:,0])
     plt.ylabel('Amplitude')
     plt.xlabel('Time')
     plt.grid()
     plt.show()
     
     fs=16000
     f, Pper_spec = signal.periodogram(Wc_matrix[0,0,:] , fs, 'flattop', scaling='spectrum')
     plt.semilogy(f, Pper_spec)
     plt.xlabel('frequency [Hz]')
     plt.ylabel('PSD')
     plt.grid()
     plt.show()
     