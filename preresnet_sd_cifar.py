import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, death_rate=0.):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.death_rate =death_rate

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
       
        if not self.training or torch.rand(1)[0] >= self.death_rate:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)

            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)
           
            if self.training:
                out /= (1. - self.death_rate)
        
            out += residual
        else:
            out = residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, death_rate=0.):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.death_rate =death_rate

    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.training or torch.rand(1)[0] >= self.death_rate:
            out = self.bn1(x)
            out = self.relu(out)
            out = self.conv1(out)
    
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv2(out)

            out = self.bn3(out)
            out = self.relu(out)
            out = self.conv3(out)
        
            if self.training:
                out /= (1. - self.death_rate)

            out += residual
        else:
            out = residual

        return out

class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, death_mode='linear', death_rate=0.5):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        block = Bottleneck if depth >=44 else BasicBlock
        
        nblocks = (depth - 2) // 2 
        if death_mode == 'uniform':
            death_rates = [death_rate] * nblocks
            print("Stochastic Depth: uniform mode")
        elif death_mode == 'linear':
            death_rates = [float(i + 1) * death_rate / float(nblocks)
                for i in range(nblocks)]
            print("Stochastic Depth: linear mode")
        else:
            death_rates = [0.] * (3 * n)
            print("Stochastic Depth: none mode")
 
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, death_rates[:n])
        self.layer2 = self._make_layer(block, 32, n, death_rates[n:2*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, death_rates[2*n:], stride=2)
        self.bn1 = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(64 * block.expansion, 64 * block.expansion)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, death_rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(planes * block.expansion),
                #nn.AvgPool2d((2,2), stride = (2, 2))
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, death_rate=death_rates[0]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, death_rate=death_rates[i]))

        return nn.Sequential(*layers)

    def split2(self, x):
        x1, x2 = torch.split(x,x.shape[1]/2,1)
        return x1, x2

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.bn1(x)
        x = self.relu(x)    
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)

        output = self.fc(x)
         
        
        return output


def resnet(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

