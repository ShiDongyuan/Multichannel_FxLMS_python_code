import  torch
import  numpy   as      np 
from    nptdms  import  TdmsFile

#-----------------------------------------------------------------------------------------
# Class: TDMS_reader
# Description : reading the reference and disturbance signals from TDMS file.
#-----------------------------------------------------------------------------------------
class TDMS_reader:
    
    def __init__(self, file_name='Reference_Disturbance.tdms', Num_Ref=None, Num_Err=None):
        """ Creates a new reader of tdms file.

        :param file_name: The file name of the reading tdms file.
        :param Num_Ref  : The number of the reference sensors. 
        :param Num_Err  : The number of the error sensors. 
        """
        self.tdms  = TdmsFile.read(file_name)
        self.Num_r = Num_Ref
        self.Num_e = Num_Err
    
    def reading_to_tensor(self):
        """ Reading the error signal and reference signal to tensor 

        Returns:
            Refer_vector : The reference signal tensor [Number of reference x T]
            Error_vector : The error signal tensor [Number of error x T]
        """
        tdms_file = self.tdms 
        Error_vector = []
        Refer_vector = []
        for group in tdms_file.groups():
            for ii, channel in enumerate(group.channels()):
                if ii < self.Num_r :
                    Refer_vector.append(channel[:])
                else :
                    Error_vector.append(channel[:])
        
        return torch.from_numpy(np.array(Refer_vector)).type(torch.float), torch.from_numpy(np.array(Error_vector)).type(torch.float)

if __name__=="__main__":
    
        a = TDMS_reader(Num_Ref=4, Num_Err=4)
        refer, error = a.reading_to_tensor()
        
        print(error.shape)