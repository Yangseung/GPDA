import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, gradOutput):
        self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        # self.coeff = 1.0
        return -self.coeff * gradOutput


class AlexNetFc(nn.Module):
    def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=31):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])

        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_alexnet.classifier[6].in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(model_alexnet.classifier[6].in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.use_bottleneck:
            y = self.bottleneck_layer(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}
class ResNetFc(nn.Module):
    def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        if new_cls:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
                self.bottleneck.weight.data.normal_(0, 0.005)
                self.bottleneck.bias.data.fill_(0.0)
                self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
                self.bottleneck_relu = nn.ReLU(inplace=True)
                self.bottleneck_drop = nn.Dropout(p=0.5)
                self.fc = nn.Linear(bottleneck_dim, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
                self.fc.weight.data.normal_(0, 0.01)
                self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = model_resnet.fc

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            y = self.bottleneck_drop(self.bottleneck_relu(self.bottleneck_bn(self.bottleneck(x))))
            y = self.fc(y)
        else:
            y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

class Classifier(nn.Module):
    def __init__(self, in_features=2048, bottleneck_dim=256, class_num=31):
        super(Classifier, self).__init__()
        self.bottleneck = nn.Linear(in_features, bottleneck_dim)
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck_relu = nn.ReLU(inplace=True)
        self.bottleneck_drop = nn.Dropout(p=0.5)
        self.bottleneck.weight.data.normal_(0, 0.005)
        self.bottleneck.bias.data.fill_(0.0)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)
        self.__in_features = bottleneck_dim
        self.high = 1.0
        self.low = 0.0
        self.max_iter=10000.0
        self.alpha=10
        self.iter_num=0

    def forward(self, x):
        self.iter_num += 1
        y = self.bottleneck(x)
        y = self.fc(y)
        return y

    def backward(self, gradOutput):
        self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        # self.coeff = 1.0
        return self.coeff * gradOutput


class MiddleGCN(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(MiddleGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,1,out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GCN(nn.Module):
    def __init__(self, in_feature, bias=False):
        super(GCN, self).__init__()
        # self.MiddleGCN1 = MiddleGCN(in_feature,1024)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.MiddleGCN2 = MiddleGCN(1024, 512)
        # self.bn2 = nn.BatchNorm2d(512)
        # self.MiddleGCN3 = MiddleGCN(512,256)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Parameter(torch.Tensor(in_feature, 1024))
        self.weight1 = nn.Parameter(torch.Tensor(1024, 1024))
        self.weight2 = nn.Parameter(torch.Tensor(1024, 1024))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,1024))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.high = 1.0
        self.low = 0.0
        self.max_iter = 10000.0
        self.alpha = 10
        self.iter_num = 0

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        self.iter_num += 1
        output = torch.matmul(input, self.weight)
        output = torch.matmul(adj, output)
        output = torch.matmul(adj, output)
        output = torch.matmul(adj, output)
        #output = torch.matmul(adj, output)
        output = self.relu(output)
        output = torch.matmul(output, self.weight1)
        output = torch.matmul(adj, output)
        output = torch.matmul(adj, output)
        output = torch.matmul(adj, output)
        #output = torch.matmul(adj, output)
        #output = self.relu(output)
        # output = torch.matmul(output, self.weight2)
        # output = torch.matmul(adj, output)
        # output = torch.matmul(adj, output)
        #output = torch.matmul(adj, output)
        #output = torch.matmul(adj, output)
        #output = torch.matmul(adj, output)
        if self.bias is not None:
            return self.relu(output + self.bias)
        else:
            return self.relu(output)

    def backward(self, gradOutput):
        # self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        self.coeff = 1.0
        return self.coeff * gradOutput

class AdversarialGCN(nn.Module):
    def __init__(self, in_feature, bias=False):
        super(AdversarialGCN, self).__init__()
        self.MiddleGCN1 = MiddleGCN(in_feature,1024)
        self.bn1 = nn.BatchNorm2d(1024)
        self.MiddleGCN2 = MiddleGCN(1024, 512)
        self.bn2 = nn.BatchNorm2d(512)
        self.MiddleGCN3 = MiddleGCN(512,256)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.LeakyReLU(0.2)
        self.weight = nn.Parameter(torch.Tensor(256, 1))
        self.sigmoid = nn.Sigmoid()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1,1,1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.max_iter = 10000.0
        self.alpha = 10
        self.iter_num = 0

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        self.iter_num += 1
        input = self.MiddleGCN1(input, adj)
        input = self.relu(self.bn1(input))
        input = self.MiddleGCN2(input, adj)
        input = self.relu(self.bn2(input))
        input = self.MiddleGCN3(input, adj)
        input = self.relu(self.bn3(input))
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return self.sigmoid(output + self.bias)
        else:
            return self.sigmoid(output)


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 1024)
        self.ad_layer2 = nn.Linear(1024, 1024)
        self.ad_layer3 = nn.Linear(1024, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x


class MixHopNetwork(torch.nn.Module):
    """
    MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.
    :param args: Arguments object.
    :param feature_number: Feature input number.
    :param class_number: Target class number.
    """

    def __init__(self, in_feature, class_number=1200):
        super(MixHopNetwork, self).__init__()
        self.feature_number = in_feature
        self.class_number = class_number
        self.calculate_layer_sizes()
        self.setup_layer_structure()
        self.high = 1.0
        self.low = 0.0
        self.max_iter = 10000.0
        self.alpha = 10
        self.iter_num = 0

    def calculate_layer_sizes(self):
        self.abstract_feature_number_1 = 1200
        self.abstract_feature_number_2 = 1200
        self.order_1 = 3
        self.order_2 = 3

    def setup_layer_structure(self):
        """
        Creating the layer structure (3 convolutional upper layers, 3 bottom layers) and dense final.
        """
        self.upper_layers = [SparseNGCNLayer(self.feature_number, 400, i, 0.5) for i
                             in range(1, self.order_1 + 1)]
        self.upper_layers = ListModule(*self.upper_layers)
        self.bottom_layers = [
            DenseNGCNLayer(self.abstract_feature_number_1, 400, i, 0.5) for i in
            range(1, self.order_2 + 1)]
        self.bottom_layers = ListModule(*self.bottom_layers)
        self.fully_connected = torch.nn.Linear(self.abstract_feature_number_2, self.class_number)

    def calculate_group_loss(self):
        """
        Calculating the column losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            upper_column_loss = torch.norm(self.upper_layers[i].weight_matrix, dim=0)
            loss_upper = torch.sum(upper_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            bottom_column_loss = torch.norm(self.bottom_layers[i].weight_matrix, dim=0)
            loss_bottom = torch.sum(bottom_column_loss)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def calculate_loss(self):
        """
        Calculating the losses.
        """
        weight_loss = 0
        for i in range(self.order_1):
            loss_upper = torch.norm(self.upper_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_upper
        for i in range(self.order_2):
            loss_bottom = torch.norm(self.bottom_layers[i].weight_matrix)
            weight_loss = weight_loss + self.args.lambd * loss_bottom
        return weight_loss

    def forward(self, features, normalized_adjacency_matrix):
        """
        Forward pass.
        :param normalized adjacency_matrix: Target matrix as a dict with indices and values.
        :param features: Feature matrix.
        :return predictions: Label predictions.
        """
        self.iter_num += 1
        abstract_features_1 = torch.cat(
            [self.upper_layers[i](normalized_adjacency_matrix, features) for i in range(self.order_1)], dim=1)
        abstract_features_2 = torch.cat(
            [self.bottom_layers[i](normalized_adjacency_matrix, abstract_features_1) for i in range(self.order_2)],
            dim=1)
        predictions = torch.nn.functional.relu(self.fully_connected(abstract_features_2))
        # predictions = torch.nn.functional.log_softmax(self.fully_connected(abstract_features_2), dim=1)
        return predictions

    def backward(self, gradOutput):
        # self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
        self.coeff = 1.0
        return self.coeff * gradOutput

class SparseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Sparse Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(SparseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform(self.weight_matrix)
        torch.nn.init.xavier_uniform(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.matmul(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features, p = self.dropout_rate, training = self.training)
        base_features = torch.nn.functional.relu(base_features)
        for iteration in range(self.iterations-1):
            base_features = torch.matmul(normalized_adjacency_matrix, base_features)
        return base_features

class DenseNGCNLayer(torch.nn.Module):
    """
    Multi-scale Dense Feature Matrix GCN layer.
    :param in_channels: Number of features.
    :param out_channels: Number of filters.
    :param iterations: Adjacency matrix power order.
    :param dropout_rate: Dropout value.
    """
    def __init__(self, in_channels, out_channels, iterations, dropout_rate):
        super(DenseNGCNLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.iterations = iterations
        self.dropout_rate = dropout_rate
        self.define_parameters()
        self.init_parameters()

    def define_parameters(self):
        """
        Defining the weight matrices.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(1, self.out_channels))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform(self.weight_matrix)
        torch.nn.init.xavier_uniform(self.bias)

    def forward(self, normalized_adjacency_matrix, features):
        """
        Doing a forward pass.
        :param normalized_adjacency_matrix: Normalized adjacency matrix.
        :param features: Feature matrix.
        :return base_features: Convolved features.
        """
        base_features = torch.matmul(features, self.weight_matrix)
        base_features = torch.nn.functional.dropout(base_features, p=self.dropout_rate, training=self.training)
        for iteration in range(self.iterations-1):
            base_features = torch.matmul(normalized_adjacency_matrix, base_features)
        return base_features

class ListModule(torch.nn.Module):
    """
    Abstract list layer class.
    """
    def __init__(self, *args):
        """
        Module initializing.
        """
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        """
        Getting the indexed layer.
        """
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        """
        Iterating on the layers.
        """
        return iter(self._modules.values())

    def __len__(self):
        """
        Number of layers.
        """
        return len(self._modules)