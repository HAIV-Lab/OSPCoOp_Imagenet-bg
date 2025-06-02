CUDA_VISIBLE_DEVICES=0 python train.py   --loss1 1.5 --loss2 0.5 --trainer OSPCoOp --shots 1 --eval_freq 50 --seed 1  --output_dir ./runs/1shots1

CUDA_VISIBLE_DEVICES=0 python train.py   --loss1 1.5 --loss2 0.5 --trainer OSPCoOp --shots 1 --eval_freq 50 --seed 2  --output_dir ./runs/1shots2

CUDA_VISIBLE_DEVICES=0 python train.py   --loss1 1.5 --loss2 0.5 --trainer OSPCoOp --shots 1 --eval_freq 50 --seed 3  --output_dir ./runs/1shots3