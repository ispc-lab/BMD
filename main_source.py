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
from utils.net_utils import Entropy, CrossEntropyLabelSmooth
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

def train(args, model, dataloader, criterion, optimizer, epoch_idx=0.0):
    
    model.train()
    loss_stack = []
    
    iter_idx = epoch_idx * len(dataloader)
    iter_max = args.epochs * len(dataloader)
    
    for imgs_train, imgs_test, imgs_label, imgs_idx in tqdm(dataloader, ncols=60):
        
        iter_idx += 1
        imgs_train = imgs_train.cuda()
        imgs_label = imgs_label.cuda()

        embed_feat, pred_cls = model(imgs_train)
        imgs_onehot_label = torch.zeros_like(pred_cls).scatter(1, imgs_label.unsqueeze(1), 1)
        
        loss = criterion(pred_cls, imgs_onehot_label)
        
        lr_scheduler(optimizer, iter_idx, iter_max)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())

    train_loss = np.mean(loss_stack)
    
    return train_loss

def test(args, model, dataloader, criterion):
    
    model.eval()
    loss_stack = []
    label_stack = []
    pred_stack = []
    
    for imgs_train, imgs_test, imgs_label, imgs_idx in tqdm(dataloader, ncols=60):
        
        imgs_test = imgs_test.cuda()
        _, pred_cls = model(imgs_test)
        img_onehot_labels = torch.zeros_like(pred_cls).scatter(1, imgs_label.cuda().unsqueeze(1), 1)
        
        loss = criterion(pred_cls, img_onehot_labels)
        _, pred_idx = torch.max(pred_cls.cpu(), dim=1)
        
        loss_stack.append(loss.cpu().item())
        label_stack.append(imgs_label)
        pred_stack.append(pred_idx)
        
    test_loss = np.mean(loss_stack)
    label_stack = torch.cat(label_stack, dim=0)
    pred_stack = torch.cat(pred_stack, dim=0)
    test_acc = torch.sum(label_stack == pred_stack) / (len(pred_stack) + 1e-4) * 100
    
    if args.dataset == "VisDA":
        confu_mat = confusion_matrix(label_stack, pred_stack)
        acc_list = confu_mat.diagonal()/confu_mat.sum(axis=1) * 100
        test_acc = acc_list.mean()
        acc_str = " ".join(["{:.2f}".format(i) for i in acc_list])
    else:
        acc_str = None
        
    return test_loss, test_acc, acc_str

def log_args(args):
    s = "==========================================\n"
    
    s += ("python" + " ".join(sys.argv) + "\n")
    
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    
    s += "==========================================\n"
    
    return s

def log_str(args, str):
    if args.log_file is not None:
        args.log_file.write(str + "\n")
        args.log_file.flush()
    print(str)
    
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    local_time = time.localtime()[0:5]
    this_dir = os.path.join(os.path.dirname(__file__), ".")
    
    if not args.test:
        save_dir = os.path.join(this_dir, "checkpoints_sfda", args.dataset, "source_"+str(args.s_idx), "source_checkpoint_{}_seed_{}".format(args.note, args.seed))
    else:
        save_dir = os.path.dirname(args.checkpoint)
        
    args.save_dir = save_dir
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if not args.test:
        arg_str = log_args(args)
        args.log_file = open(os.path.join(save_dir, "log_source_training.txt"), "w")
        args.log_file.write(arg_str)
        args.log_file.flush()
    else:
        return 

    model = SFDA(args)
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.cuda()
    
    if not args.without_wandb:
        wandb.init(name='traing_log_{:02d}_{:02d}_{:02d}_{:02d}_{:02d}'\
                        .format(local_time[0], local_time[1], local_time[2],
                                local_time[3], local_time[4]),
                config=args,
                project="SFDANet_{}".format(args.dataset),
                sync_tensorboard=True)
    
    param_group = []
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr*0.1}]
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    for k, v in model.class_layer.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    source_data_list = open(os.path.join(args.source_data_dir, "image_list.txt"), "r").readlines()
    target_data_list = open(os.path.join(args.target_data_dir, "image_list.txt"), "r").readlines()
    source_data_size = len(source_data_list)
    source_tr_size = int(source_data_size * 0.9)
    source_te_size = source_data_size - source_tr_size
    source_train_data_list, source_test_data_list = torch.utils.data.random_split(source_data_list, [source_tr_size, source_te_size],
                                                                    generator=torch.Generator().manual_seed(args.seed))
    
    source_train_dataset = SFDADataset(args, source_train_data_list, d_type="source")
    source_test_dataset = SFDADataset(args, source_test_data_list, d_type="source")
    
    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True, 
                                    num_workers=args.num_workers, drop_last=False)
    
    source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size*3, shuffle=False, 
                                    num_workers=args.num_workers, drop_last=False)
    
    criterion = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=0.1, reduction=True)
    
    best_source_test_acc = 0
    best_source_test_per_acc = None
    best_epoch_idx = 0
    
    notation_str =  "=================================================\n"
    notation_str += "        START TRAINING ON THE SOURCE:{}          \n".format(args.s_idx)
    notation_str += "================================================="
    log_str(args, notation_str)
    
    for epoch_idx in tqdm(range(args.start_epoch, args.epochs),ncols=60):
        
        train_loss = train(args, model, source_train_loader, criterion, optimizer, epoch_idx)
        train_loss_str = "Epoch:{}/{}\n".format(epoch_idx, args.epochs)
        train_loss_str += "train_loss:{:.3f}".format(train_loss)
        log_str(args, train_loss_str)
        
        if epoch_idx % 1 == 0:
            with torch.no_grad():
                test_loss, test_acc, acc_str = test(args, model, source_test_loader, criterion, )
            
            if acc_str is not None:
                test_result_str = "test_loss:{:.3f}\n".format(test_loss)
                test_result_str += (acc_str + "\t" + "test_acc:{:.2f}".format(test_acc))
            else:
                test_result_str = "test_loss:{:.3f} test_acc:{:.2f}".format(test_loss, test_acc)
                
            log_str(args, test_result_str)
            
            if test_acc > best_source_test_acc:
                best_epoch_idx = epoch_idx
                best_source_test_acc = test_acc
                best_source_test_per_acc = acc_str
                best_checkpoint_file = "{}_best_source_checkpoint.pth".format(args.dataset)
                torch.save({
                    "epoch":epoch_idx,
                    "model_state_dict":model.state_dict()}, os.path.join(save_dir, best_checkpoint_file))

            best_result_str = "best_epoch:{} \n".format(best_epoch_idx)
            if acc_str is not None:
                best_result_str += "best_test_acc:{:.2f}\t".format(best_source_test_acc) + best_source_test_per_acc
            else:
                best_result_str += "best_test_acc:{:.2f}\t".format(best_source_test_acc)
            
            log_str(args, best_result_str)
        
        checkpoint_file = "{}_latest_source_checkpoint.pth".format(args.dataset)
        torch.save({
            "epoch":epoch_idx,
            "model_state_dict":model.state_dict()}, os.path.join(save_dir, checkpoint_file))

        if not args.without_wandb:
            wandb.log({"train_loss":train_loss,
                       "test_loss":test_loss,
                       "test_acc":test_acc})

if __name__ == "__main__":
    
    args = build_args()
    args.log_file = None
    
    if args.dataset == "VisDA":
        args.source_data_dir = "./data/VisDA/train/"
        args.target_data_dir = "./data/VisDA/validation/"
        args.class_num = 12
        
    elif args.dataset == "OfficeHome":
        
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
    
    set_random_seed(args.seed)
    main(args)