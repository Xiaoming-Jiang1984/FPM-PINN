
import cv2
import numpy as np
from scipy.io import loadmat
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import math
import os
import torch
from torch import nn

ngpu= 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
def pupilfunction(wavelength, ps, Ny, Nx, NA): 
    dx = ps
    dy = ps
    dkx = 1 / (Nx * dx)
    dky = 1 / (Ny * dy)
    kx1D = dkx * (-np.floor(Nx / 2) + range(Nx))
    ky1D = dky * (-np.floor(Ny / 2) + range(Ny))
    [kx, ky] = np.meshgrid(np.fft.ifftshift(kx1D), np.fft.ifftshift(ky1D))
    k2 = kx ** 2 + ky ** 2
    kspaceNA = np.zeros([Ny, Nx])
    idx = np.where(k2 * np.square(wavelength) <= np.square(NA))
    kspaceNA[idx] = 1
    kspaceNA= np.fft.fftshift(kspaceNA)
    return kspaceNA
kxky_index_mat = loadmat("kx_ky_index.mat")
kxky_index=np.int32(kxky_index_mat["kxky_index"])
updateOrder = loadmat("updateorder.mat")
Image_num_index = np.squeeze(updateOrder["Image_num_index"]).tolist()
with open('./order_25_overlap_31.txt','r') as order:
    LED_order = (np.array(order.readlines(),dtype=np.int16) - 1).tolist()  
sparse_num_index = []
current_LED_index = []
for item in LED_order:
    current_LED_index.append(Image_num_index.index(item)) 
current_LED_index.sort()
for i in range(len(current_LED_index)):
    sparse_num_index.append(Image_num_index[current_LED_index[i]])
Raw_data = loadmat("RAW.mat")
RAW = Raw_data["RAW"]
RAW= RAW/np.max(RAW)
[N, M, L] = RAW.shape 
I_sum = np.sum(RAW)
LED_num_x = 15
LED_num_y = 15
Total_Led = LED_num_x * LED_num_y
LED_center = (LED_num_x * LED_num_y - 1) / 2 
print(LED_center)
NA = 0.1
Mag = 1
LED2stage = 90.88e3
LEDdelta = 4e3
Pixel_size = 1.845 / Mag
Lambda = 0.532
k = 2 * np.pi / Lambda
kmax = 1 / Lambda * NA
Mag_image = 4
Pixel_size_image = Pixel_size / 1
Pixel_size_image_freq = 1 / Pixel_size_image / (M * 1)
Nx_small = N
Ny_small = M

 
Aperture_fun = pupilfunction(Lambda, Pixel_size, M, N, NA)
Aperture_fun_intial_real =Aperture_fun
Aperture_fun_intial_imag = np.zeros([Ny_small,Nx_small],np.float32)
Aperture_fun_32 = np.float32(Aperture_fun)
Aperture_fun=np.complex64(Aperture_fun)
Aperture_fun_t = torch.from_numpy(Aperture_fun)
Aperture_fun_gpu= Aperture_fun_t.cuda()

Ny = M * Mag_image
Nx = N * Mag_image
ps = Pixel_size / Mag_image
Fcenter_X = math.floor (Nx/2)
Fcenter_Y = math.floor (Ny/2)
halfNx_small =math.floor(Nx_small/2)
halfNy_small =math.floor(Ny_small/2)
crop_Y=[]
crop_X=[]
RAW_UP=np.zeros([25,Ny,Nx],dtype=np.float32)
RAW_original=np.zeros([25,M,N],dtype=np.float32)
for idx in range(len(sparse_num_index)):
    updateIdx = sparse_num_index[idx]
    crop_Y.append( kxky_index[updateIdx, 1])
    crop_X.append( kxky_index[updateIdx, 0])
    intensity_idx= RAW[:,:,updateIdx] 
    RAW_original[idx,:,:]=intensity_idx
    IUp= cv2.resize(intensity_idx,(Ny,Nx),interpolation=cv2.INTER_NEAREST)
    RAW_UP[idx,:,:]=IUp    
RAW_original_t= torch.from_numpy(RAW_original)
RAW_original_gpu = RAW_original_t.cuda()
crop_Y=np.array(crop_Y)
crop_X=np.array(crop_X)

def generateRand(noise_level):
    rand_new=np.zeros([25,Ny,Nx],dtype=np.float32)
    for i in range(25):
        new_rand = np.random.uniform(0, noise_level, size=(Ny,Nx))
        rand_new[i,:,:]= new_rand        
    return rand_new        
def conv1_fun(in_channels,out_channels):
    return nn.Sequential(nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,padding=1),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU()                 
    )   
def conv10_fun(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.Softmax()                 
    )    
class up_fun(nn.Module):
    def __init__(self,in_channels,out_channels, output_size):
        super(up_fun, self).__init__()
        self.output_size = output_size
        self.upsample2d  = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_2d     = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.BN          = nn.BatchNorm2d(out_channels)
        self.LR          = nn.LeakyReLU()        
    def forward(self, x):                
        x = self.upsample2d(x)         
        x = self.conv_2d(x)
        x = self.BN  (x) 
        x = self.LR(x)
        return x           
def conv3_fun(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU()                 
    )
def conv_2times_fun(in_channels,out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU()                 
    )
dim =800  
class FourierPtychography_Unet(nn.Module):
    def __init__(self):
        super().__init__()       
        self.conv1 = conv1_fun(25, 64)       
        self.Maxpool1= nn.MaxPool2d(2)       
        self.conv2 = conv_2times_fun(64,128)        
        self.Maxpool2= nn.MaxPool2d(2)       
        self.conv3 = conv3_fun(128,256)       
        self.Maxpool3= nn.MaxPool2d(2)       
        self.conv4 = conv_2times_fun(256,512)        
        self.Maxpool4= nn.MaxPool2d(2)        
        self.conv5 = conv_2times_fun(512,1024)        
        self.Maxpool5= nn.MaxPool2d(2)       
        self.conv6 = up_fun(1024,512,output_size=[1, 512, int(dim/8),int(dim/8)])            
        self.conv6_12= conv_2times_fun(1024,512)  
        self.conv7 = up_fun(512,256,output_size=[1, 256, int(dim/4),int(dim/4)])       
        self.conv7_12= conv_2times_fun(512,256)      
        self.conv8 = up_fun(256,128,output_size=[1,128,int(dim/2),int(dim/2)])       
        self.conv8_12= conv_2times_fun(256,128)      
        self.conv9 = up_fun(128,64,output_size=[1,64,dim,dim])        
        self.conv9_12= conv_2times_fun(128,64)        
        self.conv10 = conv10_fun(64,2)      
    def forward(self, x,kxky_index_y,kxky_index_x,numberimage,idx):   
        conv1 = self.conv1(x)        
        Maxpool_1= self.Maxpool1(conv1)        
        conv2 = self.conv2(Maxpool_1)         
        Maxpool_2= self.Maxpool2(conv2)        
        conv3 = self.conv3(Maxpool_2)       
        Maxpool_3= self.Maxpool3(conv3)       
        conv4 = self.conv4(Maxpool_3)         
        Maxpool_4= self.Maxpool4(conv4)        
        conv5 = self.conv5(Maxpool_4)
        conv6 = self.conv6(conv5)   
        merge1=torch.cat([conv6, conv4], dim=1)
        conv6_12= self.conv6_12(merge1)           
        conv7 =self.conv7(conv6_12)   
        merge2=torch.cat([conv7, conv3], dim=1)
        conv7_12= self.conv7_12(merge2)   
        conv8 =self.conv8(conv7_12)
        merge3=torch.cat([conv8, conv2], dim=1)
        conv8_12= self.conv8_12(merge3)
        conv9 = self.conv9(conv8_12) 
        merge4 = torch.cat([conv1, conv9], dim=1)  
        conv9_12 =self.conv9_12(merge4)
        conv10= self.conv10(conv9_12)  
        amplitude = conv10[0,0,:,:,]
        phase =     conv10 [0,1,:,:]   
        amplitude=torch.multiply(amplitude,1/Mag_image/Mag_image)
        phase=    torch.add(torch.multiply(phase,np.pi),0.0)  
        if ((idx+1)%10==0):
            phase_cpu=phase.cpu().detach().numpy() 
            datamatstr = ("results/phase_%04d.mat") % (idx)
            datamatname = ("phase%04d") % (idx)
            sio.savemat(datamatstr, {datamatname: phase_cpu})            
        realpart= torch.multiply(amplitude,torch.cos(phase))
        imagpart= torch.multiply(amplitude,torch.sin(phase))     
        obj = torch.complex(realpart,imagpart)     
        obj_FFT_IFFT =torch.fft.fftshift( torch.fft.fft2(obj))               
        output=[]       
        for i in range(numberimage):
            Subspecturm  = torch.narrow(torch.narrow(obj_FFT_IFFT, 0, kxky_index_y[i], Ny_small), 1, kxky_index_x[i], Nx_small)  
            waveFFT = torch.multiply(Subspecturm,Aperture_fun_gpu)
            wave= torch.fft.ifft2(torch.fft.ifftshift(waveFFT))
            I= torch.reshape(torch.abs(wave),(1,M,N)) 
            if (i==0):
                output=I
            else:
                output =torch.cat([output,I],axis=0)               
        return output       
model =FourierPtychography_Unet()
loss_fn = torch.nn.L1Loss(reduce=True, size_average=True)
optimizer=torch.optim.SGD(model.parameters(), lr=5, momentum=0.9)
model.to(device)
numiter=1000
for epoch in range(numiter):
    noise= generateRand(1/10.0)       
    data= RAW_UP+noise
    data=np.reshape(data,[1,25,Ny,Nx])
    data_t=torch.from_numpy(data)
    data_gpu= data_t.cuda()   
    optimizer.zero_grad() 
    I = model(data_gpu,crop_Y,crop_X,25,epoch)
    loss= loss_fn(I,RAW_original_gpu)     
    print('loss %05d %f'%( epoch,loss ))
    loss.backward()
    optimizer.step()   