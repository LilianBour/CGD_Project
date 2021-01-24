import torch
from torch import nn
from torch.nn import functional
from ResNet50 import ResNet50

class L2_Normalisation(nn.Module):
    def __init__(self):
        super(L2_Normalisation, self).__init__()
    def forward(self,x):
        return functional.normalize(x,p=2,dim=-1)

class MultipleGlobalDescriptors(nn.Module):
    def __init__(self,pc):
        super(MultipleGlobalDescriptors,self).__init__()
        self.pc=pc
    def forward(self,x):
        if self.pc==1:
            return torch.flatten(functional.avg_pool2d(x, [14,14])*14*14,start_dim=1) #Sum Pooling instead of SPoC, using mean * feature map size to get the sum
            #was return x.mean(dim=[-1,-2]) #SPoC
            #source : https://stackoverflow.com/questions/50838876/how-to-perform-sum-pooling-in-pytorch
        if self.pc == float('inf'):
            return torch.flatten(functional.adaptive_max_pool2d(x,output_size=(1,1)),start_dim=1) #MAC
            #https://ro.uow.edu.au/cgi/viewcontent.cgi?article=7554&context=eispapers
        if self.pc == 3:
            return torch.flatten(functional.avg_pool2d(x.clamp(min=1e-6).pow(self.pc), (x.size(-2), x.size(-1))).pow(1./self.pc),start_dim=1)#GeM
            #was return torch.sign(x.pow(self.pc).mean(dim=[-1,-2]))*(torch.abs(x.pow(self.pc).mean(dim=[-1,-2])).pow(1.0/self.pc))
            #source : https://amaarora.github.io/2020/08/30/gempool.html#pytorch-implementation

class CGD(nn.Module):
    def __init__(self,global_descriptors,feature_dimensions,classes):
        super(CGD,self).__init__()

        # -- PART 1 : ResNet50 & Getting feature maps --
        backbone=ResNet50()
        self.feature_maps=[]
        for name, child_module in backbone.named_children():
            #We skip Linear and AvgPool because we want the feature maps
            if isinstance(child_module,nn.Linear) or isinstance(child_module,nn.AdaptiveAvgPool2d):
                continue
            else:
                #Add FM (feature maps)
                self.feature_maps.append(child_module)
        self.feature_maps = nn.Sequential(*self.feature_maps)

        # -- PART 2 : Main Module - Multiple Global Descriptors --
        a = int(feature_dimensions/len(global_descriptors))
        self.global_descriptor_list=[]
        self.main_modules_list = []
        for i in range(len(global_descriptors)):
            if global_descriptors[i]=="S":
                self.pc = 1 #pc set to 1 for SPoC
            if global_descriptors[i]=="M":
                self.pc = float('inf') #pc set to inf for MAC
            if global_descriptors[i]=="G":
                self.pc = 3 #pc fixed to 3 for GeM
            self.global_descriptor_list.append(MultipleGlobalDescriptors(pc=self.pc))
            self.main_modules_list.append(nn.Sequential(nn.Linear(2048,a,bias=False), L2_Normalisation())) #FC + L2norm (see model's fig)
        #Convert to ModuleList
        self.global_descriptor_list=nn.ModuleList(self.global_descriptor_list)
        self.main_modules_list = nn.ModuleList(self.main_modules_list)

        # -- PART 3 : Auxiliary Module - Classification Loss --
        self.aux_module= nn.Sequential(nn.BatchNorm1d(2048),nn.Linear(2048, classes, bias=True)) #BatchNorm + FC(see model's fig)

    def forward(self,x):
        GDs = []
        fm=self.feature_maps(x)
        for i in range(len(self.global_descriptor_list)):
            if i == 0: #Auxiliary module on first GD
                Cl = self.aux_module(self.global_descriptor_list[i](fm))
            GD = self.main_modules_list[i](self.global_descriptor_list[i](fm))
            GDs.append(GD)
        GDs = functional.normalize(torch.cat(GDs, dim=1), dim=-1) #Concatenate and normalize the feature vector
        return GDs, Cl

