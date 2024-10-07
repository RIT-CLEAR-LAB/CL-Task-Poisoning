from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, relu
import torchbnn as bnn  # You need to install torchbnn for Bayesian layers
from backbone import MammothBackbone, register_backbone

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.

    Args:
        in_planes: number of input channels
        out_planes: number of output channels
        stride: stride of the convolution

    Returns:
        convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BayesianBasicBlock(nn.Module):
    """
    The basic block of ResNet, modified for Bayesian convolution layers.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, prior_mu=0.0, prior_sigma=0.1) -> None:
        """
        Instantiates the basic block of the network with Bayesian layers.

        Args:
            in_planes: the number of input channels
            planes: the number of channels (to be possibly expanded)
        """
        super(BayesianBasicBlock, self).__init__()
        self.return_prerelu = False
        # Replace conv layers with Bayesian Conv2d
        self.conv1 = bnn.BayesConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = bnn.BayesConv2d(planes, planes, kernel_size=3, stride=1, padding=1, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                bnn.BayesConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, prior_mu=prior_mu, prior_sigma=prior_sigma),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, input_size)

        Returns:
            output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        if self.return_prerelu:
            self.prerelu = out.clone()

        out = relu(out)
        return out

class BayesianResNet(MammothBackbone):
    """
    ResNet network architecture with Bayesian layers. Designed for complex datasets.
    """

    def __init__(self, block: BayesianBasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, prior_mu=0.0, prior_sigma=0.1) -> None:
        """
        Instantiates the layers of the network.

        Args:
            block: the basic Bayesian ResNet block
            num_blocks: the number of blocks per layer
            num_classes: the number of output classes
            nf: the number of filters
        """
        super(BayesianResNet, self).__init__()
        self.return_prerelu = False
        self.device = "cpu"
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = bnn.BayesConv2d(3, nf * 1, kernel_size=3, stride=1, padding=1, prior_mu=prior_mu, prior_sigma=prior_sigma)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        # Replace classifier with Bayesian Linear
        self.classifier = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=nf * 8 * block.expansion, out_features=num_classes)
        self.feature_dim = nf * 8 * block.expansion

    def _make_layer(self, block: BayesianBasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.

        Args:
            block: ResNet basic block
            planes: channels across the network
            num_blocks: number of blocks
            stride: stride

        Returns:
            ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def kl_loss(self):
        """
        Compute the KL divergence loss from Bayesian layers.
        """
        kl = 0.0
        kl += self.conv1.kl_loss()
        kl += self.classifier.kl_loss()
        # Include KL loss from other layers if needed
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                kl += block.conv1.kl_loss() + block.conv2.kl_loss()
                if len(block.shortcut) > 0:
                    kl += block.shortcut[0].kl_loss()  # for shortcut connection
        return kl

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.

        Args:
            x: input tensor (batch_size, *input_shape)
            returnt: return type (a string among 'out', 'features', 'both', and 'full')

        Returns:
            output tensor (output_classes)
        """
        out_0 = self.bn1(self.conv1(x))  # 64, 32, 32
        out_0 = relu(out_0)
        if hasattr(self, 'maxpool'):
            out_0 = self.maxpool(out_0)

        out_1 = self.layer1(out_0)  # -> 64, 32, 32
        out_2 = self.layer2(out_1)  # -> 128, 16, 16
        out_3 = self.layer3(out_2)  # -> 256, 8, 8
        out_4 = self.layer4(out_3)  # -> 512, 4, 4

        feature = avg_pool2d(out_4, out_4.shape[2])  # -> 512, 1, 1
        feature = feature.view(feature.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return (out, feature)

        raise NotImplementedError("Unknown return type. Must be in ['out', 'features', 'both'] but got {}".format(returnt))

# Register the Bayesian ResNet18 model
@register_backbone("bayesian_resnet18")
def bayesian_resnet18(num_classes: int, num_filters: int = 64, prior_mu=0.0, prior_sigma=0.1) -> BayesianResNet:
    """
    Instantiates a Bayesian ResNet18 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        BayesianResNet network
    """
    return BayesianResNet(BayesianBasicBlock, [2, 2, 2, 2], num_classes, num_filters, prior_mu, prior_sigma)

# Register the Bayesian ResNet34 model
@register_backbone("bayesian_resnet34")
def bayesian_resnet34(num_classes: int, num_filters: int = 64, prior_mu=0.0, prior_sigma=0.1) -> BayesianResNet:
    """
    Instantiates a Bayesian ResNet34 network.

    Args:
        num_classes: number of output classes
        num_filters: number of filters

    Returns:
        BayesianResNet network
    """
    return BayesianResNet(BayesianBasicBlock, [3, 4, 6, 3], num_classes, num_filters, prior_mu, prior_sigma)
