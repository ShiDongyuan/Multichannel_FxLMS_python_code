import torch 
import numpy        as np 
import torch.nn     as nn 
import torch.optim  as optim
import scipy.signal as signal 

from Bcolors        import bcolors
from rich.console   import Console
from rich.table     import Column, Table
from rich.progress  import track

#------------------------------------------------------------
# Class : Normalized McFxLMS algorithm
# This is a Normalzed McFxLMS algorithm class.
#------------------------------------------------------------
class McFxNLMS_algorithm():

    def __init__(self, R_num, S_num, Len, Sec, device):
        '''
        Parameters:
        param1 - R_num : the number of the reference microphones.
        param2 - S_num : the number of the secondary source.
        param3 - Len   : the length of the control filter.
        param4 - Sec   : the secondary path matrix [S_num x E_num x Ls]. 
        '''
        self.Wc    = torch.zeros(R_num, S_num, Len, requires_grad=True, dtype=torch.float, device=device)
        self.Xd    = torch.zeros(R_num, Len, dtype=torch.float, device=device)
        self.Xf    = torch.zeros(R_num, Sec.shape[2], dtype=torch.float, device=device) # [R_num x Ls] delay line for filterd reference.
        self.Xdm   = torch.zeros(Sec.shape[2], R_num, Len, device=device) # [Ls x R_num x Len]
        self.Xfm   = torch.zeros(Len, R_num, Sec.shape[2], device=device) # [Len x R_num, Ls]
        self.Sec   = Sec.to(device)
        #----------------------------#
        # the number of configuration 
        #----------------------------#
        self.r_num = R_num
        self.s_num = S_num 
        self.e_num = Sec.shape[1]
        self.len   = Len
        self.ls    = Sec.shape[2]
        
        self.Yd    = torch.zeros(S_num, self.ls, device=device)
        #----------------------------#
        # Print the object information 
        #----------------------------#
        console = Console()
        table   = Table(show_header=True, header_style="bold magenta")
        table.add_column("Layers", style="dim", width=12,justify='center')
        table.add_column(f'[{self.r_num}x{self.s_num}x{self.e_num}] McFxNLMS algorithm', justify="left")
        table.add_column("Dimension", justify="left")
        table.add_column("Elements", justify="left")
        table.add_column("Weights", justify="left")
        
        table.add_row(
            "-1-",
            "Control filter",
            f'[{self.r_num} x {self.s_num} x {self.len}]',
            f'{self.r_num*self.s_num}',
            f'{self.r_num*self.s_num*self.len}'
        )
        table.add_row(
            "-2-",
            "Secondary path estimate",
            f'[{self.s_num} x {self.e_num} x {self.ls}]',
            f'{self.e_num*self.s_num}',
            f'{self.s_num*self.e_num*self.ls}'
        )
        
        console.print(table)
    
    def feedforward(self, xin):
        '''
        Parameter:
        param1 - xin : the reference signal [R_num x 1]
        param2 - Dis : the disturbance signal [E_num]
        '''
        # Shifting the input delay line 
        self.Xd         = torch.roll(self.Xd,1,1)
        self.Xd[:,0]    = xin
        # Shifting the input delay matrix  
        self.Xdm        = torch.roll(self.Xdm,1,0) # [Ls x R_num x Len]
        self.Xdm[0,:,:] = self.Xd 
        
        # Shifting the input for calculating the filtered input power 
        self.Xf         = torch.roll(self.Xf,1,1)
        self.Xf[:,0]    = xin 
        self.Xfm        = torch.roll(self.Xfm,1,0) # [[Len x R_num x Ls]]
        self.Xfm[0,:,:] = self.Xf
        
        xf_n            = torch.permute(self.Xfm,(1,0,2)) # [R_num x Len x Ls]
        X_rsenl         = xf_n.unsqueeze(1).unsqueeze(1) # [R_num x 1 x 1 x Len x Ls]
        Sec_rsenl       = self.Sec.unsqueeze(0).unsqueeze(3) #[1 x S_num x E_num x 1 x Ls]
        Xf_rsen         = torch.einsum('resnl,resnl->resn',X_rsenl, Sec_rsenl)
        Xp_rse          = torch.einsum('rsen, rsen->rse',Xf_rsen,Xf_rsen)
        Xp_e            = torch.einsum('rse->e',Xp_rse) + torch.tensor(1e-12, dtype=torch.float)
        
        # ----------Computational Graph route 1 -------------------------------
        # Building the control singal >>--y_mac--<<
        Xd_e          = self.Xd.unsqueeze(1) # [R_num x 1 x Len]
        y_mac         = torch.einsum('rsl,rsl->rs',Xd_e,self.Wc) # [R_num x S_num]---> Filtering the reference signal via different control fitlers 
        y_out         = torch.einsum('rs->s',y_mac) # [S_num] ----> Sum the control sigal from different microphone.
        # Cutting the computational chain 
        y_out_NAG     = y_out.detach()
        self.Yd       = torch.roll(self.Yd,1,1)
        self.Yd[:,0]  = y_out_NAG
        # Generating the control signal 
        Yd_e          = self.Yd.unsqueeze(1) # [S_num x 1 x Ls]
        y_anti_mac    = torch.einsum('sel,sel->se', Yd_e, self.Sec) # [S_num x E_num] ---> Filtering the control signal. 
        y_anti        = torch.einsum('se->e',y_anti_mac) #[E_num] --> sun the anti-noise from different speakers.
        # e            = Dis - y_anti 
        # ----------Computational Graph route 2 -------------------------------
        # Buiding the control signal under the assumpation that all control 
        # filters are same at Ls time.
        Xdm_e         = self.Xdm.unsqueeze(2) # [Ls x R_num x 1     x Len]  
        Wc_e          = self.Wc.unsqueeze(0)  # [1  x R_num x S_num x Len]
        y_e_mac       = torch.einsum('nrsl,nrsl->nrs',Xdm_e,Wc_e) # [Ls x R_num x S_num]
        y_e_out       = torch.einsum('nrs->ns',y_e_mac) # [Ls x S_num]
        # Generating the control signal
        Yd_e_AG       = y_e_out.unsqueeze(1) # [Ls x 1 x S_num]
        Yd_e_AG       = torch.permute(Yd_e_AG,(2,1,0)) # [S_num x E_num x Ls]
        y_anti_mac_AG = torch.einsum('sel,sel->se',Yd_e_AG, self.Sec)
        y_anti_AG     = torch.einsum('se->e',y_anti_mac_AG)
        # E_estimate    = Dis - y_anti_AG 
              
        return y_anti_AG, y_anti, Xp_e 
        
    def LossFunction(self, y_anti_AG, y_anti, Dis, Xp_e): 
        '''
        Parameter:
        param1 - y_anti_AG : the anti-nose signal [E_num] with auto gradient 
        param2 - y_anti    : the anti-noise signal 
        param3 - Dis       : the disturbance signal [E_num]
        '''
        e          = Dis - y_anti 
        E_estimate = Dis - y_anti_AG
        loss       = torch.sum(torch.einsum('e,e->e',2*torch.einsum('e,e->e', e, E_estimate),1/Xp_e))
        return loss, e 
    
    def _get_coeff_(self):
        return self.Wc.cpu().detach().numpy()

#------------------------------------------------------------------------------
# Function : train_McFxNLMS_algorithm() 0.00000005
# This function is used to train the the normalized McFxLMS algorithm.
#------------------------------------------------------------------------------
def train_McFxNLMS_algorithm(Model, Ref, Disturbance, Stepsize = 0.01):
    '''
    Parameter:
    param1 - Model : the instance of the multichannel FxLMS algorithm.
    param2 - Ref   : the reference signal vector [R_num x T] 
    R_num-- the number of the reference microphones. T -- the time index.
    param3 - Disturbance : the distrubance vector [E_num x T].
    param4 - Stepsize : the value of the step size. 
    '''
    #--------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} for training the McFxLMS algorithm !!!")
    #--------------------------------------------
    
    print(bcolors.WARNING + "<<-------------------------------START---------------------------------->>" + bcolors.ENDC)
    print(f'The length of the data is {Disturbance.shape[1]}.')
    
    #Stepsize = 0.00000005  
    optimizer= optim.SGD([Model.Wc], lr=Stepsize)
    
    # bar.start()
    Erro_signal = []
    len_data = Disturbance.shape[1]
    for itera in track(range(len_data),description="Processing..."):
        # Feedfoward
        xin = Ref[:,itera].to(device)
        dis = Disturbance[:,itera].to(device)
        y_anti_AG, y_anti, Xp_e = Model.feedforward(xin)
        loss,e                  = Model.LossFunction(y_anti_AG, y_anti, dis, Xp_e)
            
        # Backward 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()
        Erro_signal.append(e.cpu().numpy()[0])
        
    print(bcolors.WARNING + "<<-------------------------------END------------------------------------>>" + bcolors.ENDC)
    return Erro_signal

#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

#------------------------------------------------------------
# Function : Generating the testing bordband noise 
#------------------------------------------------------------
def Generating_boardband_noise_wavefrom_tensor(Wc_F, Seconds, fs):
    filter_len = 1024 
    bandpass_filter = signal.firwin(filter_len, Wc_F, pass_zero='bandpass', window ='hamming',fs=fs) 
    N = filter_len + Seconds*fs
    xin = np.random.randn(N)
    y   = signal.lfilter(bandpass_filter,1,xin)
    yout= y[filter_len:]
    # Standarlize 
    yout = yout/np.sqrt(np.var(yout))
    # return a tensor of [1 x sample rate]
    return torch.from_numpy(yout).type(torch.float).unsqueeze(0)

#-------------------------------------------------------------
# Function    : Disturbance_reference_generation_from_Fvector()
# Discription : Generating the distubrane and reference signal from the defined parameters
#-------------------------------------------------------------
def Disturbance_Noise_generation_from_Fvector(fs, T, f_vector, Pri_path, Sec_path):
    """
    Pri_path and Sec_path are  One dimension arraies 
    """
    # ANC platform configuration
    t     = np.arange(0,T,1/fs).reshape(-1,1)
    len_f = 1024
    b2    = signal.firwin(len_f, [f_vector[0],f_vector[1]], pass_zero='bandpass', window ='hamming',fs=fs)
    xin   = np.random.randn(len(t))
    Re    = signal.lfilter(b2,1,xin)
    Noise = Re[len_f-1:]
    # Noise = Noise/np.sqrt(np.var(Noise))
    
    # Construting the desired signal 
    Dir, Fx = signal.lfilter(Pri_path, 1, Noise), signal.lfilter(Sec_path, 1, Noise)
    
    return torch.from_numpy(Dir).type(torch.float), torch.from_numpy(Noise).type(torch.float)

#-------------------------------------------------------------------