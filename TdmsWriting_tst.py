from nptdms import TdmsWriter, ChannelObject
import numpy as np
import torch
from scipy.io import savemat
import scipy.io as sio

# with TdmsWriter('path_to_file.tdms') as tdms_writer:
#     data_array = np.linspace(0,100,100)
#     channel = ChannelObject('Group', 'Channel1', data_array)
#     channel2 = ChannelObject('Group', 'Channel2', data_array)
#     tdms_writer.write_segment([channel, channel2])

def TDMS_writting(file_name='path_to_file.tdms', Ref=None, Distur=None):
    Num_ref = Ref.shape[1]
    Num_err = Distur.shape[1]
    
    with TdmsWriter(file_name) as tdms_writer:
        for ii in range(Num_ref):
            channel = ChannelObject('Group',f'Reference_{ii}', Ref[:,ii])     
            tdms_writer.write_segment([channel])
        
        for jj in range(Num_err):
            channel = ChannelObject('Group',f'Disturbance_{jj}', Distur[:,jj])
            tdms_writer.write_segment([channel])

# data_array = numpy.expand_dims(numpy.linspace(0,100,100),axis=1)

# print(data_array.shape)

# Data_tenosr = torch.tensor(numpy.concatenate((data_array,data_array),axis=1), dtype=float)
# print(Data_tenosr)

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
 
if __name__== "__main__":
    
    Pri_mat, Sec_mat, Refer_mat, Distur_mat = Load_noise_path_from_mat()
    print(Refer_mat.numpy().shape)
    TDMS_writting(file_name='Reference_Disturbance.tdms', Ref=Refer_mat.numpy(), Distur=Distur_mat.numpy())
    pass