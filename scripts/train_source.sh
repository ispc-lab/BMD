echo "Start source model preparing on the VisDA-C Dataset"
python main_source.py --dataset VisDA --backbone resnet101 --lr 0.001 --without_wandb --note smooth_source --s_idx 0 --num_workers 8 --seed 2021 --epochs 10
