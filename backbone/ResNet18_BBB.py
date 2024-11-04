from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu
from backbone import MammothBackbone
from layers import BBBConv2d  # Assuming you have BBBConv2d saved as `bbb_layers.py`

class BasicBlockBBB(nn.Module):
    """
    The basic block of ResNet using Bayesian convolutional layers.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, priors=None) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        :param stride: stride of the convolution
        :param priors: priors for the Bayesian layer
        """
        super(BasicBlockBBB, self).__init__()
        self.conv1 = BBBConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, priors=priors)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BBBConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, priors=priors)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                BBBConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, priors=priors),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNetBBB(MammothBackbone):
    """
    ResNet network architecture using Bayesian layers.
    """

    def __init__(self, block: BasicBlockBBB, num_blocks: List[int], num_classes: int, nf: int, priors=None) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        :param priors: priors for the Bayesian layer
        """
        super(ResNetBBB, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.priors = priors
        self.conv1 = BBBConv2d(3, nf * 1, kernel_size=3, stride=1, padding=1, bias=False, priors=priors)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block: BasicBlockBBB, planes: int, num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer using Bayesian blocks.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, priors=self.priors))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :param returnt: return type (a string among 'out', 'features', 'all')
        :return: output tensor (output_classes)
        """

        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = avg_pool2d(out, out.shape[2])  
        feature = out.view(out.size(0), -1)

        if returnt == 'features':
            return feature

        out = self.linear(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feature)

        raise NotImplementedError("Unknown return type")
    
    def kl_loss(self) -> torch.Tensor:
        """
        Calculate the total KL divergence loss across all Bayesian layers.
        :return: Total KL divergence loss
        """
        kl_loss = 0.0
        for module in self.modules():
            if isinstance(module, BBBConv2d):
                kl_loss += module.kl_loss()
        return kl_loss


def resnet18_bbb(nclasses: int, nf: int = 64, priors=None) -> ResNetBBB:
    """
    Instantiates a ResNet18 network using Bayesian layers.
    :param nclasses: number of output classes
    :param nf: number of filters
    :param priors: priors for the Bayesian layer
    :return: ResNet network with Bayesian layers
    """
    return ResNetBBB(BasicBlockBBB, [2, 2, 2, 2], nclasses, nf, priors=priors)
