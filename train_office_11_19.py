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

def image_classification_predict(loader, model, sub_model, test_10crop=True, gpu=True, softmax_param=1.0):
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
            for j in range(9):
                _, predict_out = model(inputs[j])
                predict_out = sub_model(_)
                outputs.append(nn.Softmax(dim=1)(softmax_param * predict_out))
            outputs_center = model(inputs[9])
            outputs.append(nn.Softmax(dim=1)(softmax_param * outputs_center))
            softmax_outputs = sum(outputs)
            outputs = outputs_center
            if start_test:
                all_output = outputs.data.float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    else:
        iter_val = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = iter_val.next()
            inputs = data[0]
            if gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)
            _, outputs = model(inputs)
            softmax_outputs = nn.Softmax(dim=1)(softmax_param * outputs)
            if start_test:
                all_output = outputs.data.cpu().float()
                all_softmax_output = softmax_outputs.data.cpu().float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
                all_softmax_output = torch.cat((all_softmax_output, softmax_outputs.data.cpu().float()), 0)
                all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_softmax_output, predict, all_output, all_label


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
                predict_out = sub_model(_)
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
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.data.float()
                all_label = labels.data.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.data.float()), 0)
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

    transfer_criterion = loss.SAN

    loss_params = config["loss"]

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
    batch_size = data_config['source']['batch_size']
    net_config = config['network']
    # print(**net_config['params'])
    base_network = net_config['name'](**net_config['params'])
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()
    if net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1},
                              {"params": base_network.bottleneck.parameters(), "lr": 10},
                              {"params": base_network.fc.parameters(), "lr": 10}]
        else:
            parameter_list = [{"params": base_network.feature_layers.parameters(), "lr": 1},
                              {"params": base_network.fc.parameters(), "lr": 10}]
    else:
        parameter_list = [{"params": base_network.parameters(), "lr": 1}]

    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
    ad_net = network.AdversarialNetwork(in_feature=3248)
    sub_GCN = network.MixHopNetwork(in_feature=2048)
    ad_cls = network.Classifier(in_features=2048, class_num=class_num)
    gradient_reverse_layer = network.AdversarialLayer()
    print(ad_net)
    print(sub_GCN)
    print(ad_cls)
    if use_gpu:
        ad_net = ad_net.cuda()
        sub_GCN = sub_GCN.cuda()
        ad_cls = ad_cls.cuda()

    # 공부해
    parameter_list2 = []
    parameter_list1 = []
    parameter_list1.append({"params": base_network.feature_layers.parameters(), "lr": 1})
    parameter_list2.append({"params": ad_cls.parameters(), "lr": 10})
    parameter_list2.append({"params": base_network.feature_layers.parameters(), "lr": 1})
    parameter_list.append({"params": ad_net.parameters(), "lr": 10})
    parameter_list.append({"params": sub_GCN.parameters(), "lr": 10})

    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, **(optimizer_config["optim_params"]))
    optimizer1 = optim_dict[optimizer_config["type"]](parameter_list1, **(optimizer_config["optim_params"]))
    optimizer2 = optim_dict[optimizer_config["type"]](parameter_list2, **(optimizer_config["optim_params"]))
    param_lr = []
    param_lr1 = []
    param_lr2 = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    for param_group in optimizer1.param_groups:
        param_lr1.append(param_group["lr"])
    for param_group in optimizer2.param_groups:
        param_lr2.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]
    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    match_target = 0.0
    num_target = 0.0
    best_acc = 0.0
    for i in range(config['num_iterations']):
        if i % config['test_interval'] == 0:
            base_network.train(False)
            sub_GCN.train(False)
            ad_cls.train(False)
            base_network.eval()
            sub_GCN.eval()
            ad_cls.eval()
            temp_acc = image_classification_test(dset_loaders, base_network, sub_GCN, ad_cls,
                                                 test_10crop=prep_config["test_10crop"], gpu=use_gpu)
            temp_model = nn.Sequential(base_network)
            if temp_acc > best_acc:
                best_acc = temp_acc
                best_model = temp_model
            log_str = "iter: {:05d}, precision: {:.5f}".format(i, temp_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            print(log_str)
        if i % config["snapshot_interval"] == 0:
            torch.save(nn.Sequential(base_network),
                       osp.join(config["output_path"], "iter_{:05d}_model.pth.tar".format(i)))
            torch.save(nn.Sequential(ad_cls),
                       osp.join(config["output_path"], "iter_{:05d}_cls_model.pth.tar".format(i)))

        # 1 iter
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer1 = lr_scheduler(param_lr1, optimizer1, i, **schedule_param)
        optimizer2 = lr_scheduler(param_lr2, optimizer2, i, **schedule_param)
        optimizer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                                                          Variable(inputs_target), Variable(labels_source)

        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features, outputs = base_network(inputs)
        sub_outputs = ad_cls(features)

        base_network.train(True)
        ad_net.train(True)
        sub_GCN.train(True)

        target_label = nn.Softmax(dim=-1)(outputs[batch_size:, :])

        target_label_ = torch.sum(target_label, 0)
        target_label_ = target_label_ / torch.mean(target_label_)
        sudo_label = nn.Softmax(dim=-1)(sub_outputs).data.cpu().numpy()
        adj = np.zeros((sub_outputs.size(0), sub_outputs.size(0)))
        labels_source_ = labels_source.data.cpu().numpy()
        # thresholding 해보기
        # sudo_label = sudo_label * (sudo_label > 0.5)
        adj[batch_size:, :batch_size] = sudo_label[batch_size:, labels_source_[:]]
        adj = adj + adj.transpose(1, 0)
        for aa in range(0, batch_size):
            for bb in range(0, batch_size):
                if labels_source_[aa] == labels_source_[bb]:
                    adj[aa, bb] = 1.0
        np.fill_diagonal(adj, 1)
        adj = torch.from_numpy(adj).float().cuda()

        D = torch.pow(adj.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(adj, D).t(), D)
        adj = Variable(adj).detach()

        GCN_features = sub_GCN(features, adj)
        new_features = torch.cat([features, GCN_features], dim=-1)
        ad_out3, transfer_loss, _ = transfer_criterion(new_features, ad_net, gradient_reverse_layer, target_label_,
                                                       labels_source, use_gpu)

        transfer_loss += loss_params["entropy_trade_off"] * loss.EntropyLoss(nn.Softmax(dim=1)(outputs))
        classifier_loss = nn.CrossEntropyLoss()(outputs.narrow(0, 0, batch_size), labels_source)

        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()

        ad_cls.train(True)

        sub_outputs = ad_cls(features)

        sub_classifier_loss = nn.CrossEntropyLoss(weight=target_label_.detach())(
            sub_outputs.narrow(0, 0, batch_size), labels_source)
        # sub_classifier_loss += loss_params["entropy_trade_off"] * loss.EntropyLoss(nn.Softmax(dim=1)(sub_outputs))
        sub_classifier_loss.backward(retain_graph=True)
        optimizer2.step()
        optimizer2.zero_grad()


        target_label1 = nn.Softmax(dim=-1)(sub_outputs[batch_size:, :])
        __, target_label1_ = torch.max(target_label1, -1)
        target_label1_ = target_label1_.detach()

        source_center = Variable(torch.zeros((class_num, 2048))).cuda()
        target_center = Variable(torch.zeros((class_num, 2048))).cuda()
        source_num_index = Variable(torch.zeros(class_num)).cuda()
        target_num_index = Variable(torch.zeros(class_num)).cuda()
        for index, label in enumerate(labels_source):
            source_num_index[label] += 1
            source_center[label] += features[index]
        for index, label in enumerate(target_label1_):
            target_num_index[label] += 1
            target_center[label] += features[batch_size+index]

        exist_source = torch.nonzero(source_num_index)
        exist_target = torch.nonzero(target_num_index)
        source_num_index[source_num_index == 0] = 1
        target_num_index[target_num_index == 0] = 1
        source_center /= source_num_index.unsqueeze(-1)
        target_center /= target_num_index.unsqueeze(-1)
        if i == 0:
            moving_source_center = source_center
            moving_target_center = target_center
        moving_source_center[exist_source.squeeze()] = 0.7 * moving_source_center[exist_source.squeeze()] + 0.3 * \
                                                       source_center[exist_source.squeeze()]
        moving_target_center[exist_target.squeeze()] = 0.7 * moving_target_center[exist_target.squeeze()] + 0.3 * \
                                                       target_center[exist_target.squeeze()]
        coeff = np.float(2.0 * (1.0 - 0.0) / (1.0 + np.exp(-10 * i / 10000.0)) - (1.0 - 0) + 0)
        random_shift_index = 1 + np.random.randint(class_num - 1, size=1)[0]
        shift_target_center = torch.cat(
            [moving_target_center[random_shift_index:], moving_target_center[:random_shift_index]])
        shift_loss = -coeff * nn.MSELoss()(moving_source_center, shift_target_center.detach())
        center_loss = coeff * nn.MSELoss()(moving_source_center, moving_target_center.detach())
        center_loss += shift_loss
        center_loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        moving_source_center = moving_source_center.detach()
        moving_target_center = moving_target_center.detach()

        target_label1_ = target_label1_.data.cpu().numpy()
        labels_target = labels_target.cpu().numpy()
        match_target += sum(target_label1_ == labels_target)
        num_target += np.size(target_label1_)

        if i % 100 == 0:
            print('feature Discriminator loss : {}'.format(transfer_loss.data.cpu()))
            #print('GCN fearure Discriminator loss : {}'.format(transfer_loss2.data.cpu()))
            print('Sudo target acc : {}'.format(match_target/num_target * 100.0))
            #print('Feature discriminator output : {}'.format(ad_out3))
            #print('Feature discriminator output : {}'.format(ad_out31))
            match_target=0.0
            num_target=0.0

    torch.save(best_model, osp.join(config["output_path"], "best_model.pth.tar"))
    return best_acc


if __name__ == "__main__":
    print(torch.__version__)
    parser = argparse.ArgumentParser(description='Graph Adversarial Network for Partial Domain Adaptaion')
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18, 34, 50, 101, 152; AlexNet")

    # parser.add_argument('--dset', type=str, default='imagenet', help='The dataset or source dataset used')
    # parser.add_argument('--s_dset_path', type=str, default='data/imagenet-caltech/my_imagenet_1000_list.txt',
    #                    help='source dataset path')
    # parser.add_argument('--t_dset_path', type=str, default='data/imagenet-caltech/my_caltech_84_list.txt',
    #                    help='target dataset path')
    # parser.add_argument('--dset', type=str, default='office-home', help='The dataset or source dataset used')
    # parser.add_argument('--s_dset_path', type=str, default='data/office-home/my_Art.txt',
    #                     help='source dataset path')
    # parser.add_argument('--t_dset_path', type=str, default='data/office-home/my_Clipart_shared.txt',
    #                     help='target dataset path')
    # parser.add_argument('--dset', type=str, default='visda', help='The dataset or source dataset used')
    # parser.add_argument('--s_dset_path', type=str, default='/mnt/server7_hard3/[0]Data/VisDa2017/train/my_synthetic_12_list1.txt',
    #                    help='source dataset path')
    # parser.add_argument('--t_dset_path', type=str, default='/mnt/server7_hard3/[0]Data/VisDa2017/validation/my_real_6_list1.txt',
    #                    help='target dataset path')
    parser.add_argument('--dset', type=str, default='office', help='The dataset or source dataset used')
    parser.add_argument('--s_dset_path', type=str, default='data/office/my_amazon_31_list.txt',
                        help='source dataset path')
    parser.add_argument('--t_dset_path', type=str, default='data/office/my_webcam_10_list.txt',
                        help='target dataset path')

    parser.add_argument('--test_interval', type=int, default=200, help='interval of two continuous test phase')
    parser.add_argument('--snapshot_interval', type=int, default=500, help='interval of two continuous output model')
    parser.add_argument('--output_dir', type=str, default='GPDAAW', help='output directory of our model')
    args = parser.parse_args()

    config = {}
    config["num_iterations"] = 20004
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
    config["prep"] = {"test_10crop": True, "resize_size": 256, "crop_size": 224}
    config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 0.2}
    config["optimizer"] = {"type": "SGD", "optim_params": {"lr": 1.0, "momentum": 0.9,
                                                           "weight_decay": 0.0005, "nesterov": True}, "lr_type": "inv",
                           "lr_param": {"init_lr": 0.0003, "gamma": 0.001, "power": 0.75}}

    config["dataset"] = args.dset
    if config["dataset"] == "office":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 48},
                          "target": {"list_path": args.t_dset_path, "batch_size": 48},
                          "test": {"list_path": args.t_dset_path, "batch_size": 8}}
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 48},
                          "target": {"list_path": args.t_dset_path, "batch_size": 48},
                          "test": {"list_path": args.t_dset_path, "batch_size": 8}}
        config["network"]["params"]["class_num"] = 65
    elif config["dataset"] == "imagenet":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 50},
                          "target": {"list_path": args.t_dset_path, "batch_size": 50},
                          "test": {"list_path": args.t_dset_path, "batch_size": 8}}
        config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 1.0}
        config["network"]["params"]["use_bottleneck"] = False
        config["network"]["params"]["new_cls"] = False
        config["network"]["params"]["class_num"] = 1000
    elif config["dataset"] == "caltech":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 32},
                          "target": {"list_path": args.t_dset_path, "batch_size": 32},
                          "test": {"list_path": args.t_dset_path, "batch_size": 8}}
        config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 0.1}
        # config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["params"]["use_bottleneck"] = False
        config["network"]["params"]["class_num"] = 256
    elif config["dataset"] == "visda":
        config["data"] = {"source": {"list_path": args.s_dset_path, "batch_size": 32},
                          "target": {"list_path": args.t_dset_path, "batch_size": 32},
                          "test": {"list_path": args.t_dset_path, "batch_size": 8}}
        config["loss"] = {"trade_off": 1.0, "entropy_trade_off": 0.1}
        config["network"]["params"]["class_num"] = 12
    train(config)

    # 0, 1, 5, 10, 11, 12, 15, 16, 17, 22