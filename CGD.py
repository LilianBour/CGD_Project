import torch
from torch import nn
from torch.nn import functional
from ResNet50 import ResNet50

class CGD(nn.Module):
    def __init__(self,global_descriptors,feature_dimensions,classes):
        super(CGD,self).__init__()

        # -- PART 1 : ResNet50 --
        backbone=ResNet50()
        self.feature_maps=[]
        for name, child_module in backbone.named_children():
            #We skip Linear and AvgPool because we want the feature maps
            """ if type(child_module==nn.Linear) or type(child_module==nn.AdaptiveAvgPool2d):
                continue"""
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
            self.main_modules_list.append(nn.Sequential(nn.Linear(2048,a,bias=False),ADD_L2_Normalisation()))
        self.global_descriptor_list=nn.ModuleList(self.global_descriptor_list)
        self.main_modules_list = nn.ModuleList(self.main_modules_list)

        # -- PART 3 : Auxiliary Module - Classification Loss --
        self.aux_module= nn.Sequential(nn.BatchNorm1d(2048),nn.Linear(2048, classes, bias=True))#TODO

    def forward(self,x):
        GDs = []
        fm=self.feature_maps(x)
        for i in range(len(self.global_descriptor_list)):
            if i == 0: #Auxiliary module on first GD
                Cl = self.aux_module(self.global_descriptor_list[i](fm))
            GD = self.main_modules_list[i](self.global_descriptor_list[i](fm))
            GDs.append(GD)
        GDs = functional.normalize(torch.cat(GDs, dim=1), dim=-1)
        return GDs, Cl

#TODO Try to remove the need for and additional class and do the norm directly in GCD
class ADD_L2_Normalisation(nn.Module):
    def __init__(self):
        super(ADD_L2_Normalisation,self).__init__()
    def forward(self,x):
        return functional.normalize(x,p=2,dim=-1)

#TODO module maybe not needed, put everything in CGD (try later)
class MultipleGlobalDescriptors(nn.Module):
    def __init__(self,pc):
        super(MultipleGlobalDescriptors,self).__init__()
        self.pc=pc
    def forward(self,x):
        if self.pc==1:
            return x.mean(dim=[-1,-2]) #SPoC  #TODO
        if self.pc == float('inf'):
            return torch.flatten(functional.adaptive_max_pool2d(x,output_size=(1,1)),start_dim=1) #MAC  #TODO
        if self.pc == 3:
            return torch.sign(x.pow(self.pc).mean(dim=[-1,-2]))*(torch.abs(x.pow(self.pc).mean(dim=[-1,-2])).pow(1.0/self.pc))#GeM  #TODO

