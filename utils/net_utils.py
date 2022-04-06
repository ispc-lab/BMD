import torch 
import faiss
import random 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from tqdm import tqdm 
from sklearn.cluster import SpectralClustering

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def init_multi_cent_psd_label(args, model, dataloader, flag=False, flag_NRC=False, confu_mat_flag=False):
    model.eval()
    emd_feat_stack = []
    cls_out_stack = []
    gt_label_stack = []
    
    for data_train, data_test, data_label, data_idx in tqdm(dataloader, ncols=60):
        
        data_test = data_test.cuda()
        data_label = data_label.cuda()
        if flag:
            # For G-SFDA
            embed_feat, _, cls_out = model(data_test, t=1)
        else:
            embed_feat, cls_out = model(data_test)
        emd_feat_stack.append(embed_feat)
        cls_out_stack.append(cls_out)
        gt_label_stack.append(data_label)
        
    all_gt_label = torch.cat(gt_label_stack, dim=0)
    
    all_emd_feat = torch.cat(emd_feat_stack, dim=0)
    all_emd_feat = all_emd_feat / torch.norm(all_emd_feat, p=2, dim=1, keepdim=True)
    # current VISDA-C k_seg is set to 3
    topk_num = max(all_emd_feat.shape[0] // (args.class_num * args.topk_seg), 1)
        
    all_cls_out = torch.cat(cls_out_stack, dim=0)
    _, all_psd_label = torch.max(all_cls_out, dim=1)
    acc = torch.sum(all_gt_label == all_psd_label) / len(all_gt_label)
    acc_list = [acc]
    #------------------------------------------------------------#
    multi_cent_num = args.multi_cent_num
    feat_multi_cent = torch.zeros((args.class_num, multi_cent_num, args.embed_feat_dim)).cuda()
    faiss_kmeans = faiss.Kmeans(args.embed_feat_dim, multi_cent_num, niter=100, verbose=False, min_points_per_centroid=1)
    
    iter_nums = 2
    for iter in range(iter_nums):
        for cls_idx in range(args.class_num):
            if iter == 0:
                # We apply TOP-K-Sampling strategy to obtain class balanced feat_cent initialization.
                feat_samp_idx = torch.topk(all_cls_out[:, cls_idx], topk_num)[1]
            else:
                # After the first iteration, we make use of the psd_label to construct feat cent.
                # feat_samp_idx = (all_psd_label == cls_idx)
                feat_samp_idx = torch.topk(feat_dist[:, cls_idx], topk_num)[1]
                
            feat_cls_sample = all_emd_feat[feat_samp_idx, :].cpu().numpy()
            faiss_kmeans.train(feat_cls_sample)
            feat_multi_cent[cls_idx, :] = torch.from_numpy(faiss_kmeans.centroids).cuda()
            

        feat_dist = torch.einsum("cmk, nk -> ncm", feat_multi_cent, all_emd_feat) #[N,C,M]
        feat_dist, _ = torch.max(feat_dist, dim=2)  # [N, C]
        feat_dist = torch.softmax(feat_dist, dim=1) # [N, C]
            
        _, all_psd_label = torch.max(feat_dist, dim=1)
        acc = torch.sum(all_psd_label == all_gt_label) / len(all_gt_label)
        acc_list.append(acc)
        
    log = "acc:" + " --> ".join("{:.3f}".format(acc) for acc in acc_list)
    psd_confu_mat = confusion_matrix(all_gt_label.cpu(), all_psd_label.cpu())
    psd_acc_list = psd_confu_mat.diagonal()/psd_confu_mat.sum(axis=1) * 100
    psd_acc = psd_acc_list.mean()
    psd_acc_str = "{:.2f}        ".format(psd_acc) + " ".join(["{:.2f}".format(i) for i in psd_acc_list])
    
    if args.test:
        print(log)
    else:
        print(log)
        args.log_file.write(log+"\n")
        args.log_file.flush()
        
    if args.dataset == "VisDA":
        print(psd_acc_str)
    
    if flag or flag_NRC:
        # For G-SFDA or NRC
        return feat_multi_cent, all_psd_label, all_emd_feat, all_cls_out
    else:
        # For SHOT and SHOT++
        if confu_mat_flag:
            return feat_multi_cent, all_psd_label, psd_confu_mat
        else:
            return feat_multi_cent, all_psd_label


def init_psd_label_shot_icml(args, model, dataloader):
    args.distance = "cosine"
    args.epsilon = 1e-5
    args.threshold = 0.3
    start_test = True
    model.eval()
    
    with torch.no_grad():
        iter_test = iter(dataloader)
        for _ in tqdm(range(len(dataloader)), ncols=60):
            data = iter_test.next()
            inputs = data[1]
            labels = data[2]
            inputs = inputs.cuda()
            # feas = netB(netF(inputs))
            # outputs = netC(feas)
            feas, outputs = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    acc_list = [accuracy]
    
    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    acc_list.append(acc)
    
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    acc_list.append(acc)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    log_str = "acc:" + " --> ".join("{:.3f}".format(acc) for acc in acc_list)
    
    psd_confu_mat = confusion_matrix(pred_label, all_label.float().numpy())
    psd_acc_list = psd_confu_mat.diagonal()/psd_confu_mat.sum(axis=1) * 100
    psd_acc = psd_acc_list.mean()
    psd_acc_str = "{:.2f}        ".format(psd_acc) + " ".join(["{:.2f}".format(i) for i in psd_acc_list])
    
    if not args.test:
        args.log_file.write(log_str + '\n')
        args.log_file.flush()
    
    
    print(log_str+'\n')
    print(psd_acc_str)
    # return None, pred_label.astype('int')

    return None, torch.from_numpy(pred_label.astype('int')).cuda()



def EMA_update_multi_feat_cent_with_feat_simi(args, glob_multi_feat_cent, embed_feat, cls_out=None, decay=0.99):
    
    batch_size = embed_feat.shape[0]
    class_num  = glob_multi_feat_cent.shape[0]
    multi_num  = glob_multi_feat_cent.shape[1]
    
    normed_embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
    feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_embed_feat)
    feat_simi = feat_simi.flatten(1) #[N, C*M]
    feat_simi = torch.softmax(feat_simi, dim=1).reshape(batch_size, class_num, multi_num) #[N, C, M]
    
    curr_multi_feat_cent = torch.einsum("ncm, nd -> cmd", feat_simi, normed_embed_feat)
    curr_multi_feat_cent /= (torch.sum(feat_simi, dim=0).unsqueeze(2) + 1e-8)
    
    glob_multi_feat_cent = glob_multi_feat_cent * decay + (1 - decay) * curr_multi_feat_cent
    
    return glob_multi_feat_cent


class CrossEntropyLabelSmooth(nn.Module):
    
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """      

    def __init__(self, num_classes, epsilon=0.1, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets, applied_softmax=True):
        """
        Args:
            inputs: prediction matrix (after softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (batch_size, num_classes).
        """
        if applied_softmax:
            log_probs = torch.log(inputs)
        else:
            log_probs = self.logsoftmax(inputs)
        
        if inputs.shape != targets.shape:
            # this means that the target data shape is (B,)
            targets = torch.zeros_like(inputs).scatter(1, targets.unsqueeze(1), 1)
        
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).sum(dim=1)
         
        if self.reduction:
            return loss.mean()
        else:
            return loss
        
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, max_epochs=30, lambda_u=75):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch, max_epochs)