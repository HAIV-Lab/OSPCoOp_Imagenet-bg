# example of 1-shot seed-1 id augmentation and training 
CUDA_VISIBLE_DEVICES=1 python idaug.py --train_root "ImageNet_1shot_seed1" --mask_thre 0.5 --shots 1 --seed 1 --id_aug_options "1,2,3" --id_aug_times "1,1,1" --id_aug_rate "0.5,0.5,0.5"

CUDA_VISIBLE_DEVICES=1 python train.py   --loss1 1.5 --loss2 0.5 --trainer OSPCoOp --shots 1 --eval_freq 20 --config-file "configs/trainers/OSPCoOp/vit_b16_ep20.yaml" --id_aug_dir "xxx" --seed 1  --output_dir "./runs/xxxx"
