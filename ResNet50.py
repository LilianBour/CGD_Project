import torch
import torch.nn as nn #contain function to build a network -> https://pytorch.org/docs/stable/nn.html

# -- PART 1 : Building the blocks composing ResNet --
class Basic(nn.Module):
    multip=1
    def __init__(self,input,output,stride=1,downsample=None,groups=1, width=64, padding=1,normalisation_layer=None):
        super(Basic,self).__init__()
        #TODO Useful ??
        if normalisation_layer == None :
            normalisation_layer=nn.BatchNorm2d
        #TODO remove later it just raises errors
        if groups != 1 or width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if padding > 1:
            raise NotImplementedError("Dilation/Padding > 1 not supported in BasicBlock")
        # --W--self.conv1/self.downsamble  downsample when stride !=1 --W--
        #Part 1
        self.conv1=nn.Conv2d(input,output, kernel_size=3, stride=stride, padding=1, groups=1, dilation=1, bias=False)
        self.norm1=normalisation_layer(output)
        # ReLU
        self.ReLU=nn.ReLU(inplace=True)#TODO try without inplace because the default is false
        # Part 2
        self.conv2=nn.Conv2d(output,output, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False)
        self.norm2=normalisation_layer(output)
        # Assign values
        self.downsample=downsample
        self.stride=stride
    def forward(self,x):
        id = x
        if self.downsample != None:
            id = self.downsample(id)
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.ReLU(x)
        x=self.conv2(x)
        x=self.norm2(x)
        x = x + id
        x = self.ReLU(x)
        return x

class BottleNeck(nn.Module):
    multip=4
    #In this block we use a 1x1Conv to reduce the channels of the input before we perform an expansive 3x3Conv.
    #It is followed by another 1x1Conv to go back to the original size/shape
    def __init__(self,input,output,stride=1,downsample=None,groups=1, width=64, padding=1,normalisation_layer=None):
        super(BottleNeck, self).__init__()
        actual_width=int(output*(width/64))*groups #Modified this line it may not work
        # TODO Useful ??
        if normalisation_layer == None:
            normalisation_layer = nn.BatchNorm2d
        # --W--self.conv1/self.downsamble  downsample when stride !=1 --W--
        # Part 1
        self.conv1= nn.Conv2d(input, actual_width, kernel_size=1 , stride=1,bias=False)
        self.norm1=normalisation_layer(actual_width)
        # Part 2
        self.conv2= nn.Conv2d(actual_width,actual_width, kernel_size=3, stride=stride, padding=padding, groups=groups, dilation=padding, bias=False)
        self.norm2=normalisation_layer(actual_width)
        # Part 3
        self.conv3= nn.Conv2d(actual_width, output*4, kernel_size=1 , stride=1,bias=False)
        self.norm3=normalisation_layer(output*4)
        # ReLU
        self.ReLU=nn.ReLU(inplace=True)#TODO try without inplace because the default is false
        # Assign values
        self.downsample = downsample
        self.stride = stride
    def forward(self,x):
        id = x
        if self.downsample !=None:
            id=self.downsample(id)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.ReLU(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = x + id
        x = self.ReLU(x)
        return x

#Test of Basic and BottleNeck blocks
"""
from torchsummary import summary
BN = BottleNeck(256,64)
print("˅ BottleNeck ˅ ")
summary(BN, (256, 224, 224))
print("\n˅ Basic ˅")
BS = Basic(64 ,64)
summary(BS, (64, 224, 224))
"""
# -- PART 2 : Building ResNet --
class ResNet(nn.Module):
    def __init__(self,block,layers,classes=1000,groups=1,gr_width=64,stride_to_dil=None,normalisation_layer=None):
        super(ResNet,self).__init__()
        if normalisation_layer==None: normalisation_layer=nn.BatchNorm2d
        self.normalisation_layer=normalisation_layer
        self.input=64
        self.padding=1
        self.groups=groups
        self.width = gr_width
        if stride_to_dil == None:
            stride_to_dil=[False,False,False]
        #TODO delete later it's just an error
        if len(stride_to_dil)!=3:
            raise ValueError("stride_to_dil should be non or a 3-elem tuple, got : ",format(stride_to_dil))
        self.conv1 = nn.Conv2d(3, self.input, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = normalisation_layer(self.input)
        self.ReLU = nn.ReLU(inplace=True)
        self.MaxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage1 = self.Create_Layer(block, 64, layers[0])
        self.stage2 = self.Create_Layer(block, 128, layers[1], stride=2, padd=stride_to_dil[0])
        self.stage3 = self.Create_Layer(block, 256, layers[2], stride=1, padd=stride_to_dil[1])#Stride set to 1 to remove downsampling between step 3 and 4
        self.stage4 = self.Create_Layer(block, 512, layers[3], stride=2, padd=stride_to_dil[2])
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.multip, classes)
        #TODO V???V
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def Create_Layer(self,block,output,block_,stride=1,padd=False):
        downsample=None
        last_padding=self.padding
        if padd != False:
            self.padding = self.padding * stride
            stride = 1
        if stride != 1 or self.input != output * block.multip: downsample = nn.Sequential(  nn.Conv2d(self.input, output * block.multip, kernel_size=1, stride=stride, bias=False),self.normalisation_layer(output * block.multip))
        list_of_layers=[]
        list_of_layers.append(block(self.input,output,stride,downsample,self.groups,self.width,last_padding,self.normalisation_layer))
        self.input= output * block.multip
        for i in range(1,block_): list_of_layers.append(block(self.input, output,groups=self.groups,width=self.width,padding=self.padding,normalisation_layer=self.normalisation_layer))
        return nn.Sequential(*list_of_layers)
    def forward(self,x): #Modified it may not work
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.MaxPool(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.AvgPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def ResNet50(**kwargs):
    return ResNet(BottleNeck, [3, 4, 6, 3],**kwargs)

#Comparison Models (Ours vs Original)
#Thanks to https://text-compare.com we can see that we get 14x14 FM instead of 7x7 due to the removing of the downsampling between stage 3 and stage 4
"""
from torchsummary import summary
print("˅ Ours ˅ ")
summary(ResNet50(), (3, 224, 224))
print("˅ Original ˅ ")
import torchvision.models as models
summary(models.resnet50(False), (3, 224, 224))
"""