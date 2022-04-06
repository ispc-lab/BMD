import os
import torch
import argparse
from tqdm import tqdm 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def train_transform(resize_size=256, crop_size=224,):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def test_transform(resize_size=256, crop_size=224,):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

class SFDADataset(Dataset):
    
    def __init__(self, args, data_list, d_type="source"):
        
        super(SFDADataset, self).__init__()
        
        self.d_type = d_type
        self.dataset_name = args.dataset
        
        if self.d_type == "source":
            self.data_dir = args.source_data_dir
        else:
            self.data_dir = args.target_data_dir
            
        self.data_list = [item.strip().split() for item in data_list]
        
        if self.dataset_name == "OfficeHome" or self.dataset_name == "Office":
            # To speed up the data loading, we directly load the imgs to memory, and apply the resize transform to avoid large memory consumption.
            self.resize_trans = transforms.Resize((256, 256))
            print("Dataset Loading.....")
            self.img_list = [self.resize_trans(Image.open(os.path.join(self.data_dir, item[0])).convert("RGB")) for item in tqdm(self.data_list)]
            print("Dataset Loading done!")
        
        self.train_transform = train_transform()
        self.test_transform = test_transform()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, img_idx):
        
        img_f, img_label = self.data_list[img_idx]
        if self.dataset_name == "OfficeHome" or self.dataset_name == "Office":
            img = self.img_list[img_idx]
        else:
            img = Image.open(os.path.join(self.data_dir, img_f)).convert("RGB")
        
        img_label = int(img_label)
        img_train = self.train_transform(img)
        img_test = self.test_transform(img)
        
        return img_train, img_test, img_label, img_idx
    
  
if __name__ == "__main__":
    
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data_dir", type=str, default="./data/VisDA/train/")
    parser.add_argument("--target_data_dir", type=str, default="./data/VisDA/validation/")
    
    args = parser.parse_args()
    vis_source_imglist = open(os.path.join(args.source_data_dir, "image_list.txt"), "r").readlines()
    vis_target_imglist = open(os.path.join(args.target_data_dir, "image_list.txt"), "r").readlines()
    vis_source_dsize = len(vis_source_imglist)
    vis_source_tr_size = int(vis_source_dsize * 0.9)
    vis_source_va_size = vis_source_dsize - vis_source_tr_size
    vis_sr_tr_list, vis_sr_va_list = torch.utils.data.random_split(vis_source_imglist, [vis_source_tr_size, vis_source_va_size],
                                                                    generator=torch.Generator().manual_seed(2021))
    
    vis_source_train_dataset = SFDADataset(args, vis_sr_tr_list, d_type="source")
    vis_source_test_dataset = SFDADataset(args, vis_sr_va_list, d_type="source")
    vis_target_test_dataset = SFDADataset(args, vis_target_imglist, d_type="target")
    
    print("source_train_size: ", len(vis_source_train_dataset))
    print("source_test_size : ", len(vis_source_test_dataset))
    print("target_test_size : ", len(vis_target_test_dataset))
    
    vis_train_loader = DataLoader(vis_source_train_dataset, batch_size=64, shuffle=True, drop_last=False)
    
    for img_train, img_test, img_label, img_idx in tqdm(vis_train_loader):
 
        print(img_train.shape)
        print(img_label.shape)
        break
        
        