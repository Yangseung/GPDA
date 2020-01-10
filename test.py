import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network_1 as network
import loss_s_t_1 as loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable
import pdb

optim_dict = {"SGD": optim.SGD}


# torch.manual_seed(12)
# torch.cuda.manual_seed(12)
# np.random.seed(12)
# torch.backends.cudnn.deterministic = True

def image_classification_test(loader, model, GCN, sub_model, test_10crop=True, gpu=True):
    start_test = True
    if test_10crop:
        iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
        for i in range(len(loader['test0'])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0] for j in range(10)]
            labels = data[0][1]
            if gpu:
                for j in range(10):
                    inputs[j] = Variable(inputs[j].cuda())
                labels = Variable(labels.cuda())
            else:
                for j in range(10):
                    inputs[j] = Variable(inputs[j])
                labels = Variable(labels)
            outputs = []
            for j in range(10):
                _, predict_out = model(inputs[j])
                adj = torch.matmul(_, _.transpose(1, 0)).detach()
                D = torch.pow(adj.sum(1).float(), -0.5)
                D = torch.diag(D)
                adj = torch.matmul(torch.matmul(adj, D).t(), D).detach()
                GCN_features = GCN(_, adj)
                new_features = torch.cat([_, GCN_features], dim=-1)
                predict_out = sub_model(new_features)
                outputs.append(nn.Softmax(dim=1)(predict_out))
            outputs = sum(outputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            if gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels)
            _, predict_out = model(inputs)
            adj = torch.matmul(_, _.transpose(1, 0)).detach()
            D = torch.pow(adj.sum(1).float(), -0.5)
            D = torch.diag(D)
            adj = torch.matmul(torch.matmul(adj, D).t(), D).detach()
            GCN_features = GCN(_, adj)
            new_features = torch.cat([_, GCN_features], dim=-1)
            predict_out = sub_model(new_features)
            if start_test:
                all_output = predict_out.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, predict_out.data.float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    print()
    return accuracy


def train(config):
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(
        resize_size=prep_config["resize_size"],
        crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train(
        resize_size=prep_config["resize_size"],
        crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(
            resize_size=prep_config["resize_size"],
            crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test(
            resize_size=prep_config["resize_size"],
            crop_size=prep_config["crop_size"])


    dsets = {}
    dset_loaders = {}
    data_config = config['data']
    dsets['source'] = ImageList(open(data_config['source']['list_path']).readlines(), transform=prep_dict['source'])
    dset_loaders['source'] = util_data.DataLoader(dsets['source'], batch_size=data_config['source']['batch_size'],
                                                  shuffle=True, num_workers=4)
    dsets['target'] = ImageList(open(data_config['target']['list_path']).readlines(), transform=prep_dict['target'])
    dset_loaders['target'] = util_data.DataLoader(dsets['target'], batch_size=data_config['target']['batch_size'],
                                                  shuffle=True, num_workers=4)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test" + str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                                               transform=prep_dict["test"]["val" + str(i)])
            dset_loaders["test" + str(i)] = util_data.DataLoader(dsets["test" + str(i)],
                                                                 batch_size=data_config["test"]["batch_size"],
                                                                 shuffle=False, num_workers=4)

    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], batch_size=data_config["test"]["batch_size"],
                                                    shuffle=False, num_workers=4)

    class_num = config['network']['params']['class_num']
    batch_size = data_config['target']['batch_size']
    net_config = config['network']
    # print(**net_config['params'])
    base_network = net_config['name'](**net_config['params'])
    sub_GCN = network.GCN(in_feature=2048)
    ad_cls = network.Classifier(in_features=3072, class_num=class_num)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()
        sub_GCN = sub_GCN.cuda()
        ad_cls = ad_cls.cuda()
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    base = torch.load('snapshot/2048_dis1_dis2/best_base_model.pth.tar')
    sub = torch.load('snapshot/2048_dis1_dis2/best_sub_GCN_model.pth.tar')
    ad = torch.load('snapshot/2048_dis1_dis2/best_ad_cls_model.pth.tar')

    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in base.items():
        name = k[2:]  # remove module.
    new_state_dict[name] = v
    base_network.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    for k, v in sub.items():
        name = k[2:]  # remove module.
    new_state_dict[name] = v
    sub_GCN.load_state_dict(new_state_dict)

    new_state_dict = OrderedDict()
    for k, v in ad.items():
        name = k[2:]  # remove module.
    new_state_dict[name] = v
    ad_cls.load_state_dict(new_state_dict)

    base_network.eval()
    sub_GCN.eval()
    ad_cls.eval()

    temp_acc = image_classification_test(dset_loaders, base_network, sub_GCN, ad_cls, test_10crop=prep_config["test_10crop"], gpu=use_gpu)
    log_str = "precision: {:.5f}".format(temp_acc)

    print(log_str)


if __name__ == "__main__":
    print(torch.__version__)
    parser = argparse.ArgumentParser(description='Graph Adversarial Network for Partial Domain Adaptaion')
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18, 34, 50, 101, 152; AlexNet")
    parser.add_argument('--dset', type=str, default='office', help='The dataset or source dataset used')
    parser.add_argument('--s_dset_path', type=str, default='data/office/my_amazon_31_list.txt',
                        help='source dataset path')
    parser.add_argument('--t_dset_path', type=str, default='data/office/my_webcam_10_list.txt',
                        help='target dataset path')
    parser.add_argument('--test_interval', type=int, default=100, help='interval of two continuous test phase')
    parser.add_argument('--snapshot_interval', type=int, default=5000, help='interval of two continuous output model')
    parser.add_argument('--output_dir', type=str, default='2048_dis1_dis2', help='output directory of our model')
    args = parser.parse_args()

    config = {}
    config["num_iterations"] = 50004
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_path"] = "snapshot/" + args.output_dir
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.mkdir(config["output_path"])
    if "ResNet" in args.net:
        config["network"] = {'name': network.ResNetFc, 'params': {'resnet_name': args.net,
                                                                  'use_bottleneck': True, 'bottleneck_dim': 256,
                                                                  'new_cls': True}}
    elif "AlexNet" in args.net:
        config["network"] = {'name': network.AlexNetFc, 'params': {'resnet_name': args.net,
                                                                   'use_bottleneck': True, 'bottleneck_dim': 256,
                                                                   'new_cls': True}}
    config["prep"] = {"test_10crop": False, "resize_size": 256, "crop_size": 224}
    config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 0.2}
    config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9,
                                                           "weight_decay": 0.0005, "nesterov": True}, "lr_type": "inv",
                           "lr_param": {"init_lr": 0.0003, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    if config["dataset"] == "office":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 32},
                          "target": {"list_path": args.t_dset_path, "batch_size": 32},
                          "test": {"list_path": args.t_dset_path, "batch_size": 32}}
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "imagenet":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 36},
                          "target": {"list_path": args.t_dset_path, "batch_size": 36},
                          "test": {"list_path": args.t_dset_path, "batch_size": 32}}
        config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 1.0}
        config["network"]["params"]["use_bottleneck"] = False
        config["network"]["params"]["new_cls"] = False
        config["network"]["params"]["class_num"] = 1000
    elif config["dataset"] == "caltech":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 32},
                          "target": {"list_path": args.t_dset_path, "batch_size": 32},
                          "test": {"list_path": args.t_dset_path, "batch_size": 32}}
        config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 0.1}
        config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["params"]["class_num"] = 256
    train(config)

    # 0, 1, 5, 10, 11, 12, 15, 16, 17, 22