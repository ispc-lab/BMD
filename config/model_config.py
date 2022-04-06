import os 
import argparse

def build_args(dataset="VisDA"):
    
    parser = argparse.ArgumentParser("This script is used to SFDA")
    parser.add_argument("--dataset", type=str, default="VisDA")
    parser.add_argument("--backbone_arch", type=str, default="resnet101", help="restnet50, resnet101, vgg")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--s_idx", type=int, default=0)
    parser.add_argument("--t_idx", type=int, default=1)
    parser.add_argument("--distance", default="cosine", type=str)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--epochs", default=30, type=int)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--without_wandb", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--note", default="None", type=str)
    parser.add_argument("--seed", default=2021, type=int)
    parser.add_argument("--multi_cent_num", default=4, type=int)
    parser.add_argument("--topk_seg", default=3, type=int)
    parser.add_argument("--lam_psd", default=0.30, type=float)
    parser.add_argument("--lam_dym", default=0.10, type=float)
    parser.add_argument("--lam_reg", default=1.0, type=float)
    parser.add_argument("--lam_ent", default=1.0, type=float)
    args = parser.parse_args()
    # args.dataset = dataset
    
    if args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.class_num = 12
        
    elif args.dataset == "OfficeHome":
        # OfficeHome dataset need to generate img_list.
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.source_data_dir = os.path.join("./data/OfficeHome", names[args.s_idx])
        args.target_data_dir = os.path.join("./data/OfficeHome", names[args.t_idx])
        args.class_num = 65
        
    elif args.dataset == "Office":
        names = ['Amazon', 'Dslr', 'Webcam']
        args.source_data_dir = os.path.join("./data/Office", names[args.s_idx])
        args.target_data_dir = os.path.join("./data/Office", names[args.t_idx])
        args.class_num = 31
        
    else:
        raise ValueError("Wrong Dataset Name!!!")
    
    if args.test:
        args.without_wandb = True
    
    return args
    