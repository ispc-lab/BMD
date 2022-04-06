import torch 
import numpy as np 
import torch.nn as nn
from torchvision import models

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, 
            "vgg16":models.vgg16, "vgg19":models.vgg19, 
            "vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn,
            "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 

class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    # self.in_features = model_vgg.classifier[6].in_features
    self.backbone_feat_dim = model_vgg.classifier[6].in_features
  
  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, 
            "resnet50":models.resnet50, "resnet101":models.resnet101,
            "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d,
            "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.backbone_feat_dim = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class Embedding_SDA(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(Embedding_SDA, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type
        self.em = nn.Embedding(2, 256)
        
    def forward(self, x, t, s=100, all_mask=False):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out = x
        if t == 0:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            out = out * self.mask
        if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            out = out * mask
        if all_mask:
            t0 = torch.LongTensor([0]).cuda()
            t1 = torch.LongTensor([1]).cuda()
            mask0 = nn.Sigmoid()(self.em(t0) * s)
            mask1 = nn.Sigmoid()(self.em(t1) * s)
            self.mask=mask0
            out0 = out * mask0
            out1 = out * mask1
        if all_mask:
            return (out0,out1), (self.mask,mask1)
        else:
            return out, self.mask
    

class Classifier(nn.Module):
    def __init__(self, embed_dim, class_num, type="linear"):
        super(Classifier, self).__init__()
        
        self.type = type
        if type == 'wn':
            self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(embed_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class SFDA(nn.Module):
    
    def __init__(self, args):
        
        super(SFDA, self).__init__()
        self.backbone_arch = args.backbone_arch # resnet101
        self.embed_feat_dim = args.embed_feat_dim # 256
        self.class_num = args.class_num           # 12 for VisDA

        if "resnet" in self.backbone_arch:   
            self.backbone_layer = ResBase(self.backbone_arch) 
        elif "vgg" in self.backbone_arch:
            self.backbone_layer = VGGBase(self.backbone_arch)
        else:
            raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        
        self.backbone_feat_dim = self.backbone_layer.backbone_feat_dim
        
        self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
        
        self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")
    
    def get_emd_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat
    
    def forward(self, input_imgs, apply_softmax=True):
        
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        
        embed_feat = self.feat_embed_layer(backbone_feat)
        
        cls_out = self.class_layer(embed_feat)
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)
        else:
            pass
        
        return embed_feat, cls_out
   

if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_arch", type=str, default="vit")
    parser.add_argument("--embed_feat_dim", type=int, default=256)
    parser.add_argument("--class_num", type=int, default=12)
    
    args = parser.parse_args()
    
    sfda_model = SFDA(args)
    rand_data = torch.rand((10, 3, 224, 224))
    embed_feat, cls_out = sfda_model(rand_data)
    
    print(embed_feat.shape)
    print(cls_out.shape)
    print(sfda_model.backbone_layer.in_features)