import os 
import sys 
import time 
import torch
import wandb 
import numpy as np 
from tqdm import tqdm 
from model.SFDA import SFDA
from dataset.dataset_class import SFDADataset
from torch.utils.data.dataloader import DataLoader 

from config.model_config import build_args
from utils.net_utils import set_random_seed
from utils.net_utils import init_multi_cent_psd_label
from utils.net_utils import EMA_update_multi_feat_cent_with_feat_simi
from sklearn.metrics import confusion_matrix


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def train(args, model, train_dataloader, test_dataloader, optimizer, epoch_idx=0.0):
    
    loss_stack = []

    iter_idx = epoch_idx * len(train_dataloader)
    iter_max = args.epochs * len(train_dataloader)
    
    with torch.no_grad():
        model.eval()
        print("update psd label bank!")
        glob_multi_feat_cent, all_psd_label = init_multi_cent_psd_label(args, model, test_dataloader)
        
        model.train()

    for imgs_train, imgs_test, imgs_label, imgs_idx in tqdm(train_dataloader): 
        
        iter_idx += 1
        imgs_train = imgs_train.cuda()
        imgs_idx = imgs_idx.cuda() 
        
        psd_label = all_psd_label[imgs_idx]
        
        embed_feat, pred_cls = model(imgs_train)
        
        if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
            psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
        
        mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
        reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
        ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
        psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
        
        if epoch_idx >= 1.0:
            loss = ent_loss + 2.0 * psd_loss
        else:
            loss = - reg_loss + ent_loss
        
        #==================================================================#
        # SOFT FEAT SIMI LOSS
        #==================================================================#
        normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
        dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
        
        dym_psd_loss = - torch.sum(torch.log(pred_cls) * dym_label, dim=1).mean() - torch.sum(torch.log(dym_label) * pred_cls, dim=1).mean()
        
        if epoch_idx >= 1.0:
            loss += 0.5 * dym_psd_loss
        #==================================================================#
        #==================================================================#
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())
            glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(args, glob_multi_feat_cent, embed_feat, decay=0.9999)
            
    train_loss = np.mean(loss_stack)
    
    return train_loss

    
def test(args, model, test_dataloader):
    
    model.eval()
    label_stack = []
    pred_stack = []
    
    for imgs_train, imgs_test, imgs_label, imgs_idx in tqdm(test_dataloader):
        
        imgs_test = imgs_test.cuda()
        
        _, pred_cls = model(imgs_test)
        
        label_stack.append(imgs_label)
        pred_stack.append(torch.max(pred_cls.cpu(), dim=1)[1])
        
    pred_stack = torch.cat(pred_stack, dim=0)
    label_stack = torch.cat(label_stack, dim=0)
    
    overall_acc = torch.sum(pred_stack == label_stack) / float(label_stack.size()[0])
    
    if args.dataset == "VisDA":
        confu_mat = confusion_matrix(label_stack, pred_stack)
        acc_list = confu_mat.diagonal()/confu_mat.sum(axis=1) * 100
        acc = acc_list.mean()
        acc_str = " ".join(["{:.2f}".format(i) for i in acc_list])

    else:
        acc = overall_acc * 100
        acc_str = "None"
        
    if args.test:
        print(acc)
        print(acc_str)
    
    return acc, acc_str

def log_args(args):
    s = "==========================================\n"
    
    s += ("python" + " ".join(sys.argv) + "\n")
    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    
    s += "==========================================\n"
    
    return s
    
def main(args):
    
    os.environ['CUDA_VIVIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")

    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_sfda", args.dataset, "s_"+str(args.s_idx)+"_t_"+str(args.t_idx),"checkpoints_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}"\
                                        .format(local_time[0], local_time[1], local_time[2],\
                                                local_time[3], local_time[4]))
    else:
        save_dir = os.path.dirname(args.checkpoint)
        
    args.save_dir = save_dir
    args.device = device
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    model = SFDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        raise ValueError("You did't specify source model!!!")
    
    model = model.to(device)
    
    if not args.without_wandb:
        wandb.init(name='traing_log_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'\
                        .format(local_time[0], local_time[1], local_time[2],
                                local_time[3], local_time[4]),
                config=args,
                project="SFDANet_{}_DA".format(args.dataset),
                sync_tensorboard=True)

    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        if "bn" in k:
            param_group += [{'params': v, 'lr': args.lr*0.1}]
        else:
            v.requires_grad = False

    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
        
    for k, v in model.class_layer.named_parameters():
        v.requires_grad = False
    

    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
        
    target_data_list = open(os.path.join(args.target_data_dir, "image_list.txt"), "r").readlines()

    target_dataset = SFDADataset(args, target_data_list, d_type="target")
    
    target_train_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, drop_last=False)
    
    target_test_loader = DataLoader(target_dataset, batch_size=args.batch_size*3, shuffle=False, 
                                    num_workers=args.num_workers, drop_last=False)

    best_acc = 0
    best_acc_str = 0
    best_epoch_idx = 0

    if not args.test:
        arg_str = log_args(args)
        args.log_file = open(os.path.join(save_dir, "log_target_adaption.txt"), "w")
        args.log_file.write(arg_str)
        args.log_file.flush()
        
        for epoch_idx in tqdm(range(args.epochs)):

            loss = train(args, model, target_train_loader, target_test_loader, optimizer, epoch_idx)
                
            with torch.no_grad():
                acc, acc_str = test(args, model, target_test_loader)
            
            if best_acc < acc:
                best_acc = acc
                best_acc_str = acc_str
                best_epoch_idx = epoch_idx

                checkpoint_file = "{}_best_source_checkpoint.pth".format(args.dataset)

                torch.save({
                    "epoch":epoch_idx,
                    "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))
            
            log_s1 = "current acc: {:.2f} \t proc: {}/{}".format(acc, epoch_idx+1, args.epochs)
            log_s2 = "current acc: " + acc_str
            log_s3 = "   best_acc: {:.2f} \t best: {}/{}".format(best_acc, best_epoch_idx+1, args.epochs)
            log_s4 = "   best_acc: " + best_acc_str
            args.log_file.write("\n".join([log_s1, log_s2, log_s3, log_s4]))
            args.log_file.flush()
            args.log_file.write("==================================\n")
            args.log_file.flush()
            
            print("\n".join([log_s1, log_s2, log_s3, log_s4]))

            if not args.without_wandb:
                wandb.log({
                    "train_loss":loss,
                    "test_acc":acc,})
    else:
        with torch.no_grad():
            acc, acc_str = test(args, model, target_test_loader)
    
if __name__ == "__main__":
    
    args = build_args(dataset="VisDA")
    set_random_seed(args.seed)
    main(args)