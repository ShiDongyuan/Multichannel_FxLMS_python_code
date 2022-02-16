import torch
import matplotlib.pyplot as plt
import scipy.io          as sio
import numpy             as np 

from Multichannel_FxLMS_algorithm import McFxLMS_algorithm, train_fxmclms_algorithm
from scipy.io                     import savemat
from scipy                        import signal
from Read_tdms                    import TDMS_reader
from Tst_4_channel_ANC_program_McFxLMS import Load_noise_path_from_mat

#------------------------------------------------------------
# Function : Save_data_to_mat()
#------------------------------------------------------------
def Save_data_to_mat(Mat_file_name ='Tst_4channel_program.mat', **kwargs):
     mdict = {}
     for arg in kwargs:
          mdict[arg] = kwargs[arg]
     savemat(Mat_file_name, mdict)
     
if __name__ == "__main__":
    a = TDMS_reader(Num_Ref=4, Num_Err=4)
    Pri_mat, Sec_mat, Refer_mat, Distur_mat = Load_noise_path_from_mat()
    
    Sec_path = torch.permute(Sec_mat,(2,0,1)) 
    Refer, Distu= a.reading_to_tensor()
    print(Refer.shape)
    
    #--------------------------------------------
    Len_control = 512 
    num_filters = 16
    
    if torch.cuda.is_available():
          device = "cuda"
    else:
          device = "cpu"
        
    controller = McFxLMS_algorithm(R_num=4, S_num=4, Len=Len_control, Sec=Sec_path, device=device)
    Erro       = train_fxmclms_algorithm(Model=controller, Ref=Refer, Disturbance=Distu, device= device, Stepsize=0.000013)
    Wc_matrix  = controller._get_coeff_()
     
    # Saving the mat 
    Err_array = np.array(Erro)
    Save_data_to_mat(Mat_file_name ='Tst_4channel_program_McFxLMS.mat', Wc_matrix=Wc_matrix, Err_array=Err_array)
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
     
    