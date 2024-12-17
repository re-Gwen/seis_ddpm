conda activate cuda_envs
python train_ddpm.py --batch_size 10 --epochs 100 --lr 1e-4 --device 'cuda:0' --in_channels 1 --time_steps 250 --lamb 0.5
#'/home/chengzhitong/seismic_ddpm/data/006_3a3_nucns_3a2_data_DX004_p2.sgy'
#source run_ddpm.sh