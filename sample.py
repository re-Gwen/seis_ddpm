import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
from ddpm import DiffusionModel
from model import Unet
import tool
import dataset

NUM_SAMPLE=20
RAW_DATA_PATH="/home/chengzhitong/seismic_ddpm/data/006_3a3_nucns_3a2_data_DX004_p2.sgy"
def load_model(path,device,model):
    data= torch.load(path,map_location=device)
    model.load_state_dict(data['model'])
    model.to(device)
    return model
import h5py
def sample(diffusion_model: DiffusionModel,num_samples,x_cond: torch.Tensor=None,norm:bool=False):
    
    diffusion_model.model.eval()    
    fp=os.path.join(os.getcwd(),'DDPM_Unet_POS_False_RAW_liked_signal_samples')
    os.makedirs(fp,exist_ok=True)
    with torch.no_grad():
         sampled_imgs = diffusion_model.sample(
                            batch_size=num_samples,train=False,x_cond=x_cond
                        )
         #fid_list=[]
         #sample_array=[]
         mean_list=[]
         for i in range(num_samples):
             img = sampled_imgs[-1][i,0,:,:]
             img = img.detach().cpu()
             img=img.numpy()
             #freq=np.fft.fft2(img)
             mean=np.mean(img)
             mean_list.append(mean)
             if norm:
                 img=tool._norm_ndarray(img)
             #fid=calculate_fid(ori_data,img)
             #fid_list.append(fid)
             #print(f'sample{i}_fid:{fid}')
             #if (i+1)%num_samples==0:
                 #print(f'sample_fid_max:{max(fid_list)},sample_fid_mean:{sum(fid_list)/len(fid_list)}')
             #img=img/(numpy.abs(img).max()) 
             plt.figure()
             plt.pcolor(img,cmap='RdGy_r',vmin=-1,vmax=1)
             plt.ylim(plt.ylim()[::-1])
             plt.colorbar()
             #plt.gca().set_aspect(1)
             '''if np.abs(mean)>1e-2:
                 pass
             else:
                img=img'''
             plt.savefig(os.path.join(fp,'sample_RAW_LIKED_CHOOSE{i}.png'.format(i=i)),dpi=600)    
             plt.close()
             torch.cuda.empty_cache()
         print(f'mean_list:{mean_list}')

if __name__ == '__main__':
    net=Unet(image_channels=2)
    device='cuda:0'
    model=load_model(path='/home/chengzhitong/seismic_ddpm/ddpm_raw/results/DDPM_Unet_POS_False_RAW_liked_signal/checkpoints/model-3_RAW-liked.pth',device=device,model=net)
    print('Successfully loaded model')
    dl_raw=dataset.RAW_data_loader(file_name=RAW_DATA_PATH,data_num=NUM_SAMPLE,resize=False,batch_size=NUM_SAMPLE,resample=True,pad=False)
    raw_data=next(iter(dl_raw))
    raw_data=raw_data.unsqueeze(1).to(device)
    ddpm = DiffusionModel(
        model=model,
        image_size=[1050, 90],
        device=device,
        pos_index=(0, 0, 0),
        pos=False,
        pos_file_path=None,
        batch_size=1,
    )
    sample(ddpm,num_samples=NUM_SAMPLE,x_cond=raw_data,norm=False)
         
