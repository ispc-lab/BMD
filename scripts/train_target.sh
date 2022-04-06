echo "Start target model adaptation on the VisDA-C Dataset"

### Before runing this script, you need to assign the checkpoint file path first.

python main_target.py --dataset VisDA --backbone resnet101 --lr 0.001 --without_wandb --checkpoint ./checkpoints_sfda/VisDA/source_0/source_checkpoint_smooth_source_seed_2021/VisDA_latest_source_checkpoint.pth --note smooth_source --num_workers 8 --seed 2021
