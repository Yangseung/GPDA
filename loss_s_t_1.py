import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

# def EntropyLoss(input_):
#     mask = input_.ge(0.000001)
#     mask_out = torch.masked_select(input_, mask)
#     entropy = -(torch.sum(mask_out * torch.log(mask_out)))
#     return entropy / float(input_.size(0))

def EntropyLoss(predict_prob, class_level_weight=None, instance_level_weight=None, epsilon=1e-20):
    """
    entropy for multi classification

    predict_prob should be size of [N, C]

    class_level_weight should be [1, C] or [N, C] or [C]

    instance_level_weight should be [N, 1] or [N]

    :param predict_prob:
    :param class_level_weight:
    :param instance_level_weight:
    :param epsilon:
    :return:
    """
    N, C = predict_prob.size()

    if class_level_weight is None:
        class_level_weight = 1.0
    else:
        if len(class_level_weight.size()) == 1:
            class_level_weight = class_level_weight.view(1, class_level_weight.size(0))
        assert class_level_weight.size(1) == C, 'fatal error: dimension mismatch!'

    if instance_level_weight is None:
        instance_level_weight = 1.0
    else:
        if len(instance_level_weight.size()) == 1:
            instance_level_weight = instance_level_weight.view(instance_level_weight.size(0), 1)
        assert instance_level_weight.size(0) == N, 'fatal error: dimension mismatch!'

    entropy = -predict_prob * torch.log(predict_prob + epsilon)
    return torch.sum(instance_level_weight * entropy * class_level_weight) / float(N)

def SAN(features, ad_net, grl_layer, target_label_, labels_source, use_gpu=True):
    loss = 0
    batch_size = features.size(0) // 2
    loss_weight = torch.ones((features.size(0), 1)).cuda()
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    ad_out3 = ad_net(grl_layer(features))
    loss_weight[:batch_size] = target_label_[labels_source].data
    loss = nn.BCELoss(loss_weight.view(-1))(ad_out3.view(-1), dc_target.view(-1))
    return ad_out3, loss, loss_weight

def SAN_dis(features, ad_net, use_gpu=True):
    loss = 0
    batch_size = features.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    ad_out = ad_net(features)
    loss = nn.BCELoss()(ad_out.view(-1), dc_target.view(-1))
    return ad_out, loss

def GraphPDA(features, ad_net, grl_layer, target_label_, labels_source, use_gpu=True):
    loss_weight = torch.ones((64,1)).cuda()
    loss = 0
    # outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = features.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    # transpose
    adj = torch.matmul(features, features.transpose(1,0)).detach()
    # adj_max, _ = torch.max(adj, 0)
    # adj = adj / adj_max
    D = torch.pow(adj.sum(1).float(), -0.5)
    D = torch.diag(D)
    # adj[adj>=0.7]=1
    # adj[adj<0.7]=0
    adj = torch.matmul(torch.matmul(adj, D).t(), D).detach()
    ad_out3 = ad_net(grl_layer(features), adj)
    # ad_out4 = ad_net(features, adj)
    # loss_weight_temp = torch.ones(1)
    # loss_weight[32:] = loss_weight_temp.expand(32,-1)

    loss_weight[:32] = target_label_[labels_source].data
    # loss_weight[:32] = loss_weight[:32]/torch.mean(loss_weight[:32])
    loss = nn.BCELoss(loss_weight.view(-1))(ad_out3.view(-1), dc_target.view(-1))
    # loss += nn.MSELoss()(ad_out3.view(-1)[:32], Variable(1-loss_weight.view(-1)[:32]))
    return ad_out3, loss, loss_weight

def GraphPDA_dis(features, ad_net, use_gpu=True):
    loss = 0
    # outer_product_out = torch.bmm(input_list[0].unsqueeze(2), input_list[1].unsqueeze(1))
    batch_size = features.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    # transpose
    adj = torch.matmul(features, features.transpose(1, 0)).detach()
    D = torch.pow(adj.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(adj, D).t(), D).detach()
    ad_out = ad_net(features, adj)
    loss = nn.BCELoss()(ad_out.view(-1), dc_target.view(-1))
    return ad_out, loss

class n_pair_mc_loss(nn.Module):
    def __init__(self):
        super(n_pair_mc_loss, self).__init__()

    def forward(self, f, f_p):
        n_pairs = len(f)
        term1 = torch.matmul(f, torch.transpose(f_p, 0, 1))
        term2 = torch.sum(f * f_p, keepdim=True, dim=1)
        f_apn = term1 - term2
        mask = torch.ones_like(f_apn) - torch.eye(n_pairs).cuda()
        f_apn = f_apn * mask
        return torch.mean(torch.logsumexp(f_apn, dim=1))

def own_mse_loss(input, target, source_exist, target_exist, size_average=True):
    L = (input - target) ** 2
    return torch.mean(L) if size_average else torch.sum(L)