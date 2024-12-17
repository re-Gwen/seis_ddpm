import os
import pathlib
from typing import BinaryIO, List, Optional, Union
from matplotlib import pyplot as plt
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from ddpm import DiffusionModel
from pathlib import Path
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import model
import dataset
from sam2.build_sam import build_sam2
import tool
import torchvision.models as models

DATA_PATH = "/home/chengzhitong/seismic_ddpm/data/006_3a3_nucns_3a2_data_DX004_p2.sgy"
SIM_DATA_PATH="/home/chengzhitong/seismic_ddpm/data/data_25000_111_noS_cs_smooth5_5d_half_halfPT.h5"
DATA_NUM = 1000
MODEL_CFG= "sam2_hiera_l.yaml"
SAM2_CKP = '/data/sam2_hiera_large.pt'

def cycle(dl: DataLoader):  # 返回一个迭代器
    while True:
        for data in dl:
            yield data


def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    ori_tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    norm:bool=False
) -> None:
    assert len(tensor.shape) == 4
    tensor = tensor[0, 0, :, :].detach().cpu()
    ori_tensor = ori_tensor[0, 0, :, :].detach().cpu()
    plt.figure()
    if norm:
      tensor=tool._norm_tensor(data=tensor)
    else:
      tensor=tensor   
    plt.pcolor(tensor, cmap="RdGy_r",vmin=-1,vmax=1)
    plt.ylim(plt.ylim()[::-1])
    plt.title('generate data')
    plt.xticks([])
    plt.ylabel('Time (s)')
    plt.colorbar()
    # plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.savefig(fp, dpi=600)
    plt.close()


def config():
    parser = argparse.ArgumentParser()
    ###train config
    parser.add_argument("--model_name", type=str, default="Unet", help="model name")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--seed", type=int, default=515, help="random seed")
    ###model config
    parser.add_argument("--in_channels", type=int, default=1, help="input channels")
    parser.add_argument("--out_channels", type=int, default=1, help="output channels")
    parser.add_argument("--time_steps",type=int, default=1000, help="number of steps")
    parser.add_argument("--lamb",type=float, default=0.5, help="lambda")
    ###pos config
    parser.add_argument("--pos", type=bool, default=False, help="position of signal")
    ###others
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        embedding_model: torch.nn.Module,
        diffusion_model: DiffusionModel,
        results_folder: str,
        dl: DataLoader,
        tgt_dl:DataLoader,
        val_dl: DataLoader,
        device=None,
        *,
        train_batch_size: int = 8,
        train_lr: float = 1e-4,
        epochs: int = 1000,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        save_and_sample_every: int = 100,
        num_samples: int = 2,
    ):
        # self.model = model
        self.embedding_model = embedding_model
        self.diffusion_model = diffusion_model
        self.channels = 1
        self.step = 0
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.device = device

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.ckp_folder = self.results_folder / "checkpoints"
        self.ckp_folder.mkdir(exist_ok=True)
        self.img_folder = self.results_folder / "images"
        self.img_folder.mkdir(exist_ok=True)
        self.log_folder = self.results_folder / "logs"
        self.log_folder.mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_folder))
        # self.results_folder.mkdir(exist_ok=True)

        self.dl = dl
        self.val_dl = val_dl
        self.tgt_dl=tgt_dl

        self.train_epochs = epochs
        self.num_steps = len(self.dl) // train_batch_size
        self.train_num_steps = self.train_epochs * self.num_steps

        self.batch_size = train_batch_size
        self.train_lr = train_lr

        self.opt = AdamW(
            # diffusion_model.model.parameters(),
            # diffusion_model.module.model.parameters(),
            [
                {
                    "params": self.diffusion_model.module.model.parameters(),
                    "lr": train_lr,
                },
                # {"params": model.bottleneck.parameters(), "lr": 10 * args.lr},
                {"params": self.embedding_model.parameters(), 
                 "lr": train_lr / 10
                },
            ],
            lr=train_lr,
            betas=adam_betas,
            weight_decay=5e-4,
        )
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    self.opt, T_max=self.train_num_steps, eta_min=0, last_epoch=-1, verbose=True
        # )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt, T_0=20, T_mult=2, last_epoch=-1, eta_min=1e-5
        )

    def save(self, milestone: int) -> None:
        data = {
            "model": self.diffusion_model.module.model.state_dict(),
            #"model": self.diffusion_model.model.state_dict(),
            "opt": self.opt.state_dict(),
            "version": "1.0",
        }

        # torch.save(data, str(self.ckp_folder / f"model-{milestone}_RAW.pth"))
        # torch.save(data, str(self.ckp_folder / f"model-{milestone}.pth"))
        torch.save(data, str(self.ckp_folder / f"model-{milestone}_RAW-liked.pth"))

    def valid(self) -> None:
        self.diffusion_model.model.eval()
        with torch.inference_mode():
            total_loss = 0.0
            for _ in range(10):
                data = next(cycle(self.val_dl)).to(self.device)
                data = data.unsqueeze(1)
                loss = self.diffusion_model(data)
                total_loss += loss.item()
            print("the validtion loss is: ", total_loss / 10)

    def train_wo_cond(
        self,
    ):
        for epoch in trange(self.train_epochs):
            loss_list = []
            self.diffusion_model.model.train()
            for idx, data in enumerate((self.dl)):
                data = data.unsqueeze(1).to(self.device)
                # data=data.to(self.device)
                # print(data.shape)
                self.opt.zero_grad()
                loss = self.diffusion_model(data,cond=False)
                if epoch == 0 and idx == 0:
                    print("the first loss is:", loss.item())
                loss.backward()
                self.opt.step()
                loss_list.append(loss.item())
                if (idx + 1) % len(self.dl) == 0:
                    print("loss:", sum(loss_list) / len(self.dl))
                    self.writer.add_scalar("loss", sum(loss_list) / len(self.dl), epoch)
                if (idx + 1) % len(self.dl) == 0 and (
                    epoch + 1
                ) % self.save_and_sample_every == 0:
                    self.diffusion_model.model.eval()
                    with torch.inference_mode():
                        milestone = (epoch + 1) // self.save_and_sample_every
                        sampled_imgs = self.diffusion_model.sample(
                            batch_size=self.num_samples,
                            train=True,
                            cond=False,
                        )
                    # for ix, sampled_img in enumerate(sampled_imgs):
                    save_image(
                        sampled_imgs[-1],
                        data,
                        #str(self.img_folder / f"sample-{milestone}_RAW.png"),
                        fp=str(self.img_folder / f"sample-{milestone}_SIM.png"),
                    )
                    self.save(milestone)
                    torch.cuda.empty_cache()
            # self.lr_scheduler.step()
    def train_w_cond(
        self,
    ):
        for epoch in trange(self.train_epochs):
            loss_list = []
            self.diffusion_model.module.model.train()
            for idx, (data,tgt_data) in enumerate(zip(self.dl,self.tgt_dl)):
                data = data.unsqueeze(1).to(self.device)
                tgt_data=tgt_data.unsqueeze(1).to(self.device)

                #n,_,_,_=tgt_data.shape

                #r=torch.rand((n,1,1,1)).to(self.device)
                #Noise=torch.randn_like(tgt_data).to(self.device)
                #tgt_data=tgt_data*r+Noise*(1-r**2)**.5

                tgt_data=tgt_data.repeat(1,3,1,1)
                '''with torch.no_grad():
                    output=self.embedding_model.forward_image(tgt_data)
                    backbone_FPN_list=output['backbone_fpn']'''
                output=self.embedding_model.forward_image(tgt_data)
                backbone_FPN_list=output
                #backbone_FPN_list=output['backbone_fpn']

                backbone_FPN_list=[tool._norm_tensor(backbone_FPN) for backbone_FPN in backbone_FPN_list]

                torch.cuda.empty_cache()

                self.opt.zero_grad()
                loss = self.diffusion_model(data,cond=True,cond_list=backbone_FPN_list)
                if epoch == 0 and idx == 0:
                    print("the first loss is:", loss.item())
                loss.backward()
                self.opt.step()
                # self.lr_scheduler.step()
                loss_list.append(loss.item())
                if (idx + 1) % len(self.dl) == 0:
                    print("loss:", sum(loss_list) / len(self.dl))
                    self.writer.add_scalar("loss", sum(loss_list) / len(self.dl), epoch)
                if (idx + 1) % len(self.dl) == 0 and (
                    epoch + 1
                ) % self.save_and_sample_every == 0:
                    self.diffusion_model.module.model.eval()
                    with torch.inference_mode():
                        milestone = (epoch + 1) // self.save_and_sample_every
                        sampled_imgs = self.diffusion_model.module.sample(
                            batch_size=self.num_samples,
                            train=True,
                            cond=True,
                            cond_list=backbone_FPN_list
                        )
                    # for ix, sampled_img in enumerate(sampled_imgs):
                    save_image(
                        sampled_imgs[-1],
                        data,
                        #str(self.img_folder / f"sample-{milestone}_RAW.png"),
                        fp=str(self.img_folder / f"sample-{milestone}_RAW_liked.png"),
                        norm=True,
                    )
                    self.save(milestone)
                    torch.cuda.empty_cache()
            self.lr_scheduler.step()
    def train_w_cond_concat(
        self,
    ):
        for epoch in trange(self.train_epochs):
            loss_list = []
            self.diffusion_model.module.model.train()
            #for idx, (data,tgt_data) in enumerate(zip(self.dl,self.tgt_dl)):
            for idx, data in enumerate(self.dl):
                data = data.unsqueeze(1).to(self.device)
                #print(data.shape)
                for tgt in self.tgt_dl:
                    tgt_data =tgt
                    break  
                tgt_data=tgt_data.unsqueeze(1).to(self.device)
                #n,_,_,_=tgt_data.shape

                #r=torch.rand((n,1,1,1)).to(self.device)
                #Noise=torch.randn_like(tgt_data).to(self.device)
                #tgt_data=tgt_data*r+Noise*(1-r**2)**.5
                torch.cuda.empty_cache()

                self.opt.zero_grad()
                loss = self.diffusion_model(data,x_cond=tgt_data,cond=False,concat=True)
                if epoch == 0 and idx == 0:
                    print("the first loss is:", loss.item())
                loss.backward()
                self.opt.step()
                # self.lr_scheduler.step()
                loss_list.append(loss.item())
                if (idx + 1) % len(self.dl) == 0:
                    print("loss:", sum(loss_list) / len(self.dl))
                    self.writer.add_scalar("loss", sum(loss_list) / len(self.dl), epoch)
                if (idx + 1) % len(self.dl) == 0 and (
                    epoch + 1
                ) % self.save_and_sample_every == 0:
                    self.diffusion_model.module.model.eval()
                    with torch.inference_mode():
                        milestone = (epoch + 1) // self.save_and_sample_every
                        sampled_imgs = self.diffusion_model.module.sample(
                            batch_size=self.num_samples,
                            train=True,
                            cond=False,
                            x_cond=tgt_data,
                            concat=True,
                        )
                    # for ix, sampled_img in enumerate(sampled_imgs):
                    save_image(
                        sampled_imgs[-1],
                        data,
                        #str(self.img_folder / f"sample-{milestone}_RAW.png"),
                        fp=str(self.img_folder / f"sample-{milestone}_RAW_liked.png"),
                        norm=True,
                    )
                    self.save(milestone)
                    torch.cuda.empty_cache()
            self.lr_scheduler.step()
    def train_w_cond_interpolate(
        self,
    ):
        for epoch in trange(self.train_epochs):
            loss_list = []
            self.diffusion_model.module.model.train()
            for idx, (data,tgt_data) in enumerate(zip(self.dl,self.tgt_dl)):
            #for idx, data in enumerate(self.dl):
                data = data.unsqueeze(1).to(self.device)
                #print(data.shape) 
                tgt_data=tgt_data.unsqueeze(1).to(self.device)
                #n,_,_,_=tgt_data.shape

                #r=torch.rand((n,1,1,1)).to(self.device)
                #Noise=torch.randn_like(tgt_data).to(self.device)
                #tgt_data=tgt_data*r+Noise*(1-r**2)**.5
                torch.cuda.empty_cache()

                self.opt.zero_grad()
                loss = self.diffusion_model(data,x_cond=tgt_data,cond=False,interpolate=True)
                if epoch == 0 and idx == 0:
                    print("the first loss is:", loss.item())
                loss.backward()
                self.opt.step()
                # self.lr_scheduler.step()
                loss_list.append(loss.item())
                if (idx + 1) % len(self.dl) == 0:
                    print("loss:", sum(loss_list) / len(self.dl))
                    self.writer.add_scalar("loss", sum(loss_list) / len(self.dl), epoch)
                if (idx + 1) % len(self.dl) == 0 and (
                    epoch + 1
                ) % self.save_and_sample_every == 0:
                    self.diffusion_model.module.model.eval()
                    with torch.inference_mode():
                        milestone = (epoch + 1) // self.save_and_sample_every
                        sampled_imgs = self.diffusion_model.module.sample(
                            batch_size=self.num_samples,
                            train=True,
                            cond=False,
                            x_cond=None,
                            interpolate=True,
                        )
                    # for ix, sampled_img in enumerate(sampled_imgs):
                    save_image(
                        sampled_imgs[-1],
                        data,
                        #str(self.img_folder / f"sample-{milestone}_RAW.png"),
                        fp=str(self.img_folder / f"sample-{milestone}_RAW_liked.png"),
                        norm=True,
                    )
                    self.save(milestone)
                    torch.cuda.empty_cache()
            self.lr_scheduler.step()

def main():
    args = config()
    ###set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    ######set cudnn
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    ###set device
    device = torch.device(args.device)
    ###model init
    #model_embedding=build_sam2(MODEL_CFG, SAM2_CKP,device=device)
    model_embedding=model.backbone('resnet18').to(device)
    model_unet = model.Unet(image_channels=args.in_channels)
    #model_unet_cond=model.Unet_condition(image_channels=args.in_channels,Condition_dim=[64,128,256])
    #model_unet_cond=model_unet_cond.to(device)
    #model_unet_cond_dp=torch.nn.parallel.DataParallel(model_unet_cond,device_ids=[0,1])
    #model_unet = model_unet.to(device)
    #model_unet_dp=torch.nn.parallel.DataParallel(model_unet,device_ids=[0,1])
    
    print("time_steps:",args.time_steps)
    ddpm = DiffusionModel(       
        #model=model_unet,
        model=model_unet,
        image_size=[1050, 90],
        device=device,
        pos_index=(0, 0, 0),
        pos=False,
        pos_file_path=None,
        timesteps=args.time_steps,
        batch_size=args.batch_size,
    )
    ddpm=ddpm.to(device)
    ddpm_dp=torch.nn.parallel.DataParallel(ddpm,device_ids=[0,])
    ###load data
    dl_RAW = dataset.RAW_data_loader(
        file_name=DATA_PATH,
        data_num=DATA_NUM,
        resize=False,
        batch_size=args.batch_size,
        resample=True,
        pad=False,
    )
    dl_SIM = dataset.load_sim_data(
        file_path=SIM_DATA_PATH, device=device, batch_size=args.batch_size, pos=args.pos,data_num=DATA_NUM,sub=True
    )
    ### res_dir
    #res_dir = f"./results/DDPM_{args.model_name}_POS_{args.pos}_RAW_signal"
    #res_dir = f"./results/DDPM_{args.model_name}_POS_{args.pos}_SIM_signal"
    res_dir = f"./results/DDPM_{args.model_name}_POS_{args.pos}_RAW_liked_signal_interpolate_lamb{args.lamb}"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    ###init trainer
    trainer_ddpm = trainer(
        #model=model_unet,
        #model=model_unet_cond,
        embedding_model=model_embedding,
        diffusion_model=ddpm_dp,
        results_folder=res_dir,
        #dl=dl_RAW,
        dl=dl_SIM,
        tgt_dl=dl_RAW,
        val_dl=None,
        device=device,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        epochs=args.epochs,
        save_and_sample_every=args.epochs // 5,
        num_samples=args.batch_size,
    )
    ####train
    #trainer_ddpm.train_wo_cond()
    #trainer_ddpm.train_w_cond()
    #trainer_ddpm.train_w_cond_concat()
    trainer_ddpm.train_w_cond_interpolate()


if __name__ == "__main__":
    main()
