import cv2
import numpy as np
import os
import pywt
import torch
from matplotlib import pyplot as plt
import numbers
import sys
import h5py
from  sklearn import metrics as mr
from skimage.metrics import structural_similarity as ssim
import segyio
import torchvision.transforms as transforms
#import cv2
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
if torch.cuda.is_available():
    torch.cuda.set_device(0)
def from_sxy(data):#289 torch.Size([25, 179, 3000])
    assert len(data.shape)==5
    _3_d_data_list=[]
    for i in range(data.shape[0]):
       for j in range(data.shape[1]):
          #_3_d_data_list.append(torch.tensor(data[i,j,:,:,:]).cuda())
          _3_d_data_list.append(torch.tensor(data[i,j,:,:,:]))
    return _3_d_data_list
def spilt_data(data,rate):
    #assert len(data.shape)==4
    train_num=int(rate*data[0].shape[len(data[0].shape)-1])
    rest_num=int((1-rate)*data[0].shape[len(data[0].shape)-1])
    vaild_num=int(train_num+rest_num/2)
    #test_num=rest_num/2
    train_list=[data[i][:,:,:train_num] for i in range(len(data))]
    vaild_list=[data[i][:,:,train_num:vaild_num] for i in range(len(data))]  
    test_list= [data[i][:,:,vaild_num:] for i in range(len(data))] 
    train_tensor=torch.stack(tensors=train_list,dim=0) 
    vaild_tensor=torch.stack(tensors=vaild_list,dim=0)
    test_tensor=torch.stack(tensors=test_list,dim=0)
    #return train_tensor,vaild_tensor,test_tensor
    return train_list,vaild_list,test_list
def from_rxy(data):
    _samples_from_rx=[]
    _samples_from_ry=[]
    for i in range(len(data)):#(25,179,1800)
        for j in range(data[0].shape[0]):
            _samples_from_rx.append(data[i][j,:,:])
    for i in range(len(data)):
        for j in range(data[0].shape[1]):
            _samples_from_ry.append(data[i][:,j,:])
    return _samples_from_rx,_samples_from_ry 
def _compute_n_patches(i_h, i_w, p_h, p_w,s_h,s_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.
    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image width
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    s_h : int
        the moving step in the image height
    s_w: int
        the moving step in the image width
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    extraction_step：moving step
    """
    n_h = np.floor((i_h - p_h)/s_h)+1
    n_w = np.floor((i_w - p_w)/s_w)+1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _compute_total_patches(h, w, p_h, p_w,s_h,s_w,aug_times=[],scales=[],max_patches=None):
    num = 0
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        num += _compute_n_patches(h_scaled, w_scaled, p_h, p_w,s_h,s_w,max_patches=None)*(aug_times+1)
    return num
def progress_bar(temp_size, total_size,patch_num,file,file_list):
    done = int(50 * temp_size / total_size)
#    sys.stdout.write("\r[%s%s][%s%s] %d%% %s" % (i+1,len(file_list),'#' * done, ' ' * (50 - done), 100 * temp_size / total_size,patch_num))
    sys.stdout.write("\r[%s/%s][%s%s] %d%% %s" % (file+1,file_list,'#' * done, ' ' * (50 - done), 100 * temp_size / total_size,patch_num))
    sys.stdout.flush()
def data_aug(img, mode=None):
    # data augmentation
    if mode == 0:
        # original
        return img
    if mode == 1:
        # flip up and down
        return np.flipud(img)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(img)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(img))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(img, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(img, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(img, k=3))
def gen_patches(data,patch_size =(64,64),stride = (32,32),file = 1,file_list = 1,total_patches_num=None,train_data_num=float('inf'),patch_num = 0,aug_times=[],scales = [],q = None,single_patches_num=None,verbose=None):
    '''
    Args:
        aug_time(list): Corresponding function data_aug, if aug_time=[],mean don`t use the aug
        scales(list): data scaling; default scales = [],mean that the data don`t perform scaling,
                      if perform scaling, you can set scales=[0.9,0.8,...]
    '''
    # read data
    h, w = data.shape
    p_h,p_w = patch_size
    s_h,s_w = stride
    ##############
    patches = []
    num = 0
    for s in scales:
        h_scaled, w_scaled = int(h*s),int(w*s)
        data_scaled = data
        for i in range(0, h_scaled-p_h+1, s_h):
            for j in range(0, w_scaled-p_w+1, s_w):
                x = data_scaled[i:i+p_h, j:j+p_w]              
                if sum(sum(x)) != 0 and x.std() > 1e-5 and x.shape==patch_size:
                    num += 1
                    patch_num += 1
                    patches.append(x)
                    if verbose:
                        progress_bar(num,total_patches_num,patch_num,file,file_list)
                    if patch_num>=train_data_num:
                        return patches,patch_num
                    if len(aug_times)!=None:
                     for _ in range(0,len(aug_times)):
                        x_aug = data_aug(x, mode=np.random.randint(0,8))
                        num += 1
                        patch_num += 1
                        patches.append(x_aug)
                        if verbose:
                            progress_bar(num,total_patches_num,patch_num,file,file_list)
                        if patch_num>=train_data_num:
                            return patches,patch_num
                elif verbose:
                    num = num+1+aug_times
                    progress_bar(num,total_patches_num,patch_num,file,file_list)      
    return patches
def plot_patches(patches): 
       plotDir_1=f'_dataset_show'
       if not os.path.exists(plotDir_1):
          os.makedirs(plotDir_1)
       for j in range(40,80):
            plt.close()    
            plt.figure()
            img=torch.transpose(patches[j],0,1).detach()
            plt.pcolor(img,cmap='seismic')
            plt.ylim(plt.ylim()[::-1])
            plt.colorbar()
            plt.gca().set_aspect(1)
            plt.savefig(f'{plotDir_1}_patches__{j}.jpg',dpi=600)
            plt.close()
def resize(patches_rx,patches_ry):
    patches_list_tensor=torch.cat([torch.stack(patches_rx[i],dim=0) for i in range(len(patches_rx))],dim=0)
    patches_list_tensor_1=torch.cat([torch.stack(patches_ry[i],dim=0) for i in range(len(patches_ry))],dim=0)
    patches_tensor=torch.cat([patches_list_tensor,patches_list_tensor_1],dim=0)
    return patches_tensor 

def data_generator(data,patch_size,stride,sclaes):
    assert isinstance(patch_size,tuple) is True
    assert isinstance(stride,tuple) is True
    patches_list=[gen_patches(data[i],patch_size,stride,scales=sclaes) for i in range(len(data))]
    return patches_list,len(patches_list)
def _process(data,norm=False,p_norm=False,resize=False):   
    #data=data.unsqueeze(0)
    #input_shape=[1050,90]
    #print(data.shape)
    if isinstance(data,np.ndarray):
        data=data.transpose()
    else:
        data=data.numpy().astype(np.float32)
        data=data.transpose()
    #data=data.astype(np.float32)
    if resize:
        data=cv2.resize(data,(1050,90))
    if norm:
        data = _norm_ndarray(data)
        #data=data/np.abs(data).max()
    if p_norm:
        data = data/(np.abs(data+1e-10))**.5
         #data=data*5.1       
    return torch.tensor(data,dtype=torch.float32)
       # data = (data)/(np.abs(data)).max()
    #return torch.tensor(data,device=device)
def mk_4_channels(data,device):
    assert len(data.shape)==2
    data=data.astype(np.float32)
    data=data.transpose()
    #normlaize
    data = data/(np.abs(data)).max()
    data = data/(np.abs(data+1e-10))**.5
    data_2=cv2.resize(data,(data.shape[1]*2,data.shape[0]*2))
    coeffs = pywt.dwt2(data_2, 'haar')
    cA, (cH, cV, cD) = coeffs
    assert cA.shape==cH.shape==cV.shape==cD.shape==data.shape
    data_4_chan=np.concatenate((data[None,...],cH[None,...],),axis=0)
    return torch.tensor(data_4_chan,device=device)
    
    
import random
def get_patches(img,
                patch_size:list,
                seed:int):
    assert len(img.shape)==2
    #random.seed(seed)
    w,h=img.shape
    array_w=np.arange(0,w-patch_size[1],1)
    array_h=np.arange(0,h-patch_size[0],1)
    trace_index=random.choice(array_w)
    time_index=random.choice(array_h)
    patch=img[trace_index:trace_index+patch_size[1]:1,time_index:time_index+patch_size[0]:1].transpose(1,0)
    return patch
def get_patches_from_trace(img,
                patch_size:list,
                seed:int):
    assert len(img.shape)==2
    #random.seed(seed)
    #img=img.permute(1,0)
    w,h=img.shape
    #print(w,h)
    assert w<h
    array_w=np.arange(0,w-patch_size[1],1)
    #array_h=np.arange(0,h-patch_size[0],1)
    trace_index=random.choice(array_w)
    #time_index=random.choice(array_h)
    patch=img[trace_index:trace_index+patch_size[1]:1,0:patch_size[0]:1].transpose(1,0)
    torch.cuda.empty_cache()
    return patch  
def get_patches_from_times(img,
                patch_size:list,
                seed:int):
    assert len(img.shape)==2
    #random.seed(seed)
    w,h=img.shape
    #array_w=np.arange(0,w-patch_size[1],1)
    array_h=np.arange(0,h-patch_size[0],1)
    #trace_index=random.choice(array_w)
    time_index=random.choice(array_h)
    patch=img[0:patch_size[1]:1,time_index:time_index+patch_size[0]:1].transpose(1,0)
    return patch
def pos_index(file_path,D_index) :
    File=h5py.File(file_path,'r')
    #data=File['r'][:]
    data_pos=File['pos'][:]
    y_pos_list=[]
    index_list=[]
    for i in range(data_pos.shape[0]):
       for j in range(data_pos.shape[1]):
           for k in range(data_pos.shape[2]):
          #_3_d_data_list.append(torch.tensor(data[i,j,:,:,:]).cuda())
              y_pos_list.append(torch.tensor(data_pos[i,j,k,:],dtype=torch.float32))
              index_list.append((i,j,k))
    y_pos=np.array(y_pos_list).reshape(len(y_pos_list),-1)
    pos_array=np.min(y_pos,axis=1)
    pos_array_sort=np.unique(np.sort(pos_array))
    D=pos_array_sort[D_index]
    l=[]
    for i in range(y_pos.shape[0]):
      if pos_array[i]<=D:
         l.append( index_list[i])
    #print(l)
    for item in l:
        if  item in index_list:
            index_list.remove(item) 
    return index_list,len(index_list)
 
def pos_index_min_max(file_path,D_index_max,D_index_min) :
    File=h5py.File(file_path,'r')
    #data=File['r'][:]
    data_pos=File['pos'][:]
    y_pos_list=[]
    index_list=[]
    for i in range(data_pos.shape[0]):
       for j in range(data_pos.shape[1]):
           for k in range(data_pos.shape[2]):
          #_3_d_data_list.append(torch.tensor(data[i,j,:,:,:]).cuda())
              y_pos_list.append(torch.tensor(data_pos[i,j,k,:],dtype=torch.float32))
              index_list.append((i,j,k))
    y_pos=np.array(y_pos_list).reshape(len(y_pos_list),-1)
    pos_array=np.min(y_pos,axis=1)
    pos_array_sort=np.unique(np.sort(pos_array))
    D_max=pos_array_sort[D_index_max]
    D_min=pos_array_sort[D_index_min]
    l=[]
    Pos=[]
    for i in range(y_pos.shape[0]):
      if pos_array[i]<=D_max and pos_array[i]>=D_min:
         l.append( index_list[i])
         Pos.append(pos_array[i])
    #print(l)
    return l
def pos_index_random_1(file_path,):
    File=h5py.File(file_path,'r')
    #data=File['r'][:]
    data_pos=File['pos'][:]
    pos_index=[data_pos[random.choice(np.arange(data_pos.shape[0])),
                         random.choice(np.arange(data_pos.shape[1])),
                        random.choice(np.arange(data_pos.shape[2])),
                        i]
                for i in range(data_pos.shape[3])]
    index=[(random.choice(np.arange(data_pos.shape[0])),
                 random.choice(np.arange(data_pos.shape[0])),
                 random.choice(np.arange(data_pos.shape[0])),
                 i) for i in range(data_pos.shape[3])]
    pos_index=torch.tensor(pos_index,dtype=torch.float32).unsqueeze(0).repeat(1050,1)
    return pos_index,index 
def caculate_sim(target,input,metrics_type='cosine'):
    #print(input.shape,target.shape)
    assert len(target.shape)==len(input.shape)==4
    assert target.shape[0]==input.shape[0]==1
    target=target[0,0,:,:]
    input=input[0,0,:,:]#batch_size==1
    #assert input.device==target.device, f'the dev of target is {target.device},the dev of input is {input.device}'
    if metrics_type=='cosine':
        sim=torch.nn.functional.cosine_similarity(target.cpu().reshape(-1),
                                                  input.cpu().reshape(-1),
                                                  dim=0).item()
    if metrics_type=='Mutual Information':
        sim=mr.normalized_mutual_info_score(target.cpu().detach().numpy(),
                                            input.cpu().detach().numpy())
    if metrics_type=='structural similarity':
        sim=ssim(target.cpu().detach().numpy(),input.cpu().detach().numpy(),data_range=2,)
    return sim
def choice_target_using_pos(D_min: float, D_max: float,file_path:str):
    _,l=pos_index_min_max(file_path,D_index_min=D_min,D_index_max=D_max)
    random_idx=random.randint(0,len(l)-1)
    return l[random_idx]
def get_dict_from_label(file_path,NUM_classes)   :
    File=h5py.File(file_path,'r')
    #data=File['r'][:]
    data_pos=File['pos'][:]
    y_pos_list=[]
    index_list=[]
    for i in range(data_pos.shape[0]):
       for j in range(data_pos.shape[1]):
           for k in range(data_pos.shape[2]):
          #_3_d_data_list.append(torch.tensor(data[i,j,:,:,:]).cuda())
              y_pos_list.append(torch.tensor(data_pos[i,j,k,:],dtype=torch.float32))
              index_list.append((i,j,k))
    y_pos=np.array(y_pos_list).reshape(len(y_pos_list),-1)
    pos_array=np.min(y_pos,axis=1)
    pos_array_sort=np.unique(np.sort(pos_array))
    list_=[(len(pos_array_sort)//NUM_classes)*i for i in range(1,NUM_classes+1)]
    list_.insert(0,0)
    list_.remove(list_[-1])
    dist={i+19:pos_index_min_max(file_path=file_path,D_index_max=i+19,D_index_min=i) 
       for i in list_}
    return dist
def load_model(path,device,model:torch.nn.Module):
    data= torch.load(path,map_location=device)
    model.load_state_dict(data['model'])
    model.to(device)
    return model
from torch.utils.data import DataLoader
#from FFT_analysis import load_model
def generator(filename,i0=0,i1=999999999999,maxDR=3000,isData=True):
    with segyio.open(filename, ignore_geometry=True,mode='r') as segyfile:
        #text_header = segyfile.text[0]
        rx0 = -9999999999
        ry0 = -9999999999
        sx0 = -9999999999
        sy0 = -9999999999
        dataL =[]
        rxL = []
        ryL = []
        sxL = []
        syL = []
        iL = []
       #print('starting')
        #print(segyfile.tracecount)
        for trace_index in range(i0,min(segyfile.tracecount,i1)):
            if isData:
                trace = segyfile.trace[trace_index]
            header=segyfile.header[trace_index]
            rx = header[segyio.TraceField.GroupX]
            ry = header[segyio.TraceField.GroupY]
            sx = header[segyio.TraceField.SourceX]
            sy = header[segyio.TraceField.SourceY]
            delta = header[segyio.TraceField.TRACE_SAMPLE_INTERVAL]/1e6
            d=((rx-rx0)**2+(ry-ry0)**2)**0.5
            #print(d,trace_index)
            if d>=maxDR:
                if len(dataL)>0:
                    yield np.array(dataL),np.array(rxL),np.array(ryL),np.array(sxL),np.array(syL),np.array(iL),delta
                dataL =[]
                rxL = []
                ryL = []
                sxL = []
                syL = []
                iL  = []
            if isData:
                dataL.append(trace.data[:])
            else:
                dataL.append(0)
            rxL.append(rx)
            ryL.append(ry)
            sxL.append(sx)
            syL.append(sy)
            iL.append(trace_index)
            rx0 = rx
            ry0 = ry
            sx0 = sx
            sy0 = sy
def process_pos(pos_data,device):
    pos_data=pos_data/10000
    assert len(pos_data.shape)==1 ,f'the shape of pos_data is {pos_data.shape}'
    pos_data=torch.tensor(pos_data[None,...],device=device,dtype=torch.float32).repeat(1050,1)
    return pos_data
def mk_time_data(data,delta):
    max_idx=np.max(data)
    x,_=np.where(data==max_idx)
    a_1 = delta*x
    d = delta
    n = data.shape[-1]
# 生成等差数列
    time_data = np.asarray([a_1 + (i-1) * d for i in range(1, n+1)]).reshape(1, -1)
    return time_data
def _norm_tensor(data:torch.Tensor):
    '''
    transform to N(0,1)
    '''  
    std_=torch.std(data)
    return (data-torch.mean(data))/(std_+1e-4) 
def _norm_ndarray(data:np.ndarray):

   # print(data.shape,type(data))
    std_=np.std(data)
    return (data-np.mean(data))/(std_+1e-7)
def transform(data,train=True):
        preprocess = transforms.Compose([
                  transforms.Resize((256,256)),
                  transforms.CenterCrop(224),])
        if train:
           if len(data.shape)==3:
                data=data.unsqueeze(1)
           else:
               pass
           data=preprocess(data)
           data_1=data*1.0
           img=torch.cat((data,data_1,data_1),dim=1)
           return img
           #print(data.shape)
        else:
            data=data.unsqueeze(1)
            data=preprocess(data)
            data_1=data*1.0
            img=torch.cat((data,data_1,data_1),dim=1)
            return img
if __name__ =='__main__':
    gen=generator(filename='/home/jiangyr/data/006_3a3_nucns_3a2_data_DX004_p2.sgy')
    (data,rx,ry,sx,sy,i,delta)=next(gen)
    print(sx,sy)
    print(delta)
    print(data.shape)

   # all_patches=_compute_n_patches(25,3000,25,25,25,25,1000)
    #all_patches_1=_compute_n_patches(179,3000,64,64,32,32,1000)
    #print(all_patches,all_patches_1)
    '''global plotDir
    plotDir = f'_show/'
    file_path="/home/jiangyr/data/data_10000_36_5d.h5"
    file_path_smooth="/home/jiangyr/data/data_10000_36_smooth30_5d.h5"
    file_path_denoise="/home/jiangyr/data/data_10000_36_5d_denoise.h5"
    File=h5py.File(file_path_denoise,'r')
    data=File['r']
    data=from_sxy(data)
    #a,b,c=spilt_data(data,0.6)
    ry,rx=from_rxy(data)
    #print(len(rx),len(ry))
    patch=get_patches(_process(ry[120],device='cpu',norm=True),64,seed=None)
    plt.figure()
    img=torch.transpose(patch,0,1).detach()
    plt.pcolor(img,cmap='seismic')
    plt.ylim(plt.ylim()[::-1])
    plt.colorbar()
   #plt.gca().set_aspect(1)
    plt.savefig(f'{plotDir}_patches_64*64.jpg',dpi=600)
    plt.close()
    #print(ry[0].shape,patch.shape)
    #data_2d=f[0]
    #patches=gen_patches(data=data_2d,patch_size=(25,25),stride=(25,25),verbose=False,scales=[1])
    #print(num,patches[0].shape)
    
    
    #patches_list,a=data_generator(e,(25,25),(25,25),sclaes=[1])
    #patches_list_1,b=data_generator(f,(25,25),(25,25),sclaes=[1])
    #print(a,len(patches_list[0]),patches_list[0][0].shape)
    #plot_patches(resize(patches_list,patches_list_1))
    #print(resize(patches_list,patches_list_1).shape)'''
    '''plt.close()
    plt.figure()
    img=torch.transpose(patches[100],0,1).detach()
    plt.pcolor(img,cmap='seismic')
    plt.ylim(plt.ylim()[::-1])
    plt.colorbar()
   #plt.gca().set_aspect(1)
    plt.savefig(f'{plotDir}_patches_100.jpg',dpi=600)
    plt.close()'''
