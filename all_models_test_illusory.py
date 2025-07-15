from matplotlib import axis
# from seaborn import heatmap
import torch
import yaml
# import cv2
import os
# from PIL import Image
import torchvision.transforms as transforms
import torchvision
# from matplotlib import pyplot as plt
# import numpy as np
from data_other_model_edge import fashion_MNIST_AG, AGMNIST_all
import icpnet_model
import icpnet_model
import random_seed
# from visdom import Visdom
# import torch.nn as nn
import time
from datetime import datetime
# import copy
# import timm
# import numpy as np
# import sys


def inference(illname):
    # inference stage
    length = len(test_dataloader)
    total_imgs = test_dataset.__len__()
    t_time = 0.
    correct_top1 = 0
    correct_top5 = 0
    correct = 0
    loss_test = 0.
    net_1.eval()
    print_epoch = 30
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, start=0):
            if i % print_epoch == 0:
                # log.write('process %3d/%3d images' % (i+1, length) + '\n')
                print('process %3d/%3d images' % (i, length))

            b, _, h, w = data['images'].shape  # mnist: b, _, h, w = data['images'].shape    cifar10: b, _, h, w = data[0][0].shape

            x_2 = torch.zeros((b, 128, h//4, w//4), device='cuda:0')  # high_mnist  high_cifar  high_fashion_mnist
            x_3 = torch.zeros((b, 256, h//8, w//8), device='cuda:0')
            x_4 = torch.zeros((b, 512, h//16, w//16), device='cuda:0')
            states = [x_2, x_3, x_4]

            images = data['images'].to(device)
            cls_labels = data['labels'].to(device)
            # img_ori = data['img_ori']
            start_time = time.time()

            prediction, edge_map = net_1(images, states)  # ours
            # prediction = net_1(images)

            duration = time.time() - start_time
            t_time += duration

            loss_test += criterion(prediction, cls_labels, flag='test')

            _, pred_cls_top1 = torch.max(prediction.data, 1)
            correct_classes = pred_cls_top1.eq(cls_labels.data).cpu().sum()
            correct_top1 += correct_classes
            for j in range(b):
                _, pred_cls_top5 = torch.sort(prediction.data, 1, True)
                pred_cls_top5_ = pred_cls_top5[j][0:5]  # shape: (bs, num_classes)
                labels_ = cls_labels[j].expand_as(pred_cls_top5_)
                correct = pred_cls_top5_.eq(labels_.data).cpu().sum()
                correct_top5 += correct

            # if i % print_epoch == 0:
                # vis_img.images(feature_[0].unsqueeze(1)*100, nrow=int(128/8), win='features_stem', opts={'title': 'features_stem'})
                # vis_img.images(feature_stem[0].unsqueeze(1), nrow=int(64/8), win='features_stem', opts={'title': 'features_stem'})
                # vis_img.images(features_[1][0].unsqueeze(1), nrow=int(128/8), win='features_s1', opts={'title': 'features_s1'})
                # vis_img.images(features_[2][0].unsqueeze(1), nrow=int(256/8), win='features_s2', opts={'title': 'features_s2'})
                # vis_img.images(features_[3][0].unsqueeze(1), nrow=int(512/8), win='features_s3', opts={'title': 'features_s3'})
                # vis_img.images(features_[4][0].unsqueeze(1), nrow=int(), win='features_s4', opts={'title': 'features_s4'})
                # vis_img.images(features_[0].unsqueeze(1).sigmoid()*255, nrow=int(64/8), win='features_ccnet', opts={'title': 'features_ccnet'})
                # vis_img.image(img_ori[0], win='image_ori', opts={'title': 'image_ori'})
                # vis_img.image(images[0].cpu().detach().numpy()*255, win='image_1', opts={'title': 'image_1'})
                # vis_img.image(img_ori[0], win='AG_image' + illname, opts={'title': 'AG_image' + illname})
                # vis_img.image(edge_map[0], win='AG_edgemap' + illname, opts={'title': 'AG_edge' + illname})
                # print(f'predict correctly: {correct_classes}')

        acc_top1 = correct_top1 / total_imgs * 100.
        acc_top5 = correct_top5 / total_imgs * 100.
        format_str = '%s, ill_name: %s, acc_TOP1: %.4f, acc_TOP5: %.4f, loss_test: %.3f, avg_time: %.3f, avg_FPS:%.3f'
        log.write(format_str % (datetime.now(), illname, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time) + '\n')
        print(format_str % (datetime.now(), illname, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time))


if __name__ == '__main__':
    # vis_img = Visdom(env='visual_explanation')
    # load configures
    file_id = open('./cfgs_edge2.yaml', 'r', encoding='UTF-8')
    cfgs = yaml.load(file_id, Loader=yaml.FullLoader)
    file_id.close()

    name = cfgs['name']['fanshion_mnist_AG']
    log = open('./'+'1test_illu_' + name + '########' + '.txt', 'a+')

    random_seed.setup_seed(3407)
    net_1 = icpnet_model.ICPNet(num_classes=10)
    # net_1 = torchvision.models.vgg16(num_classes=10)
    # net_1 = torchvision.models.resnet18(num_classes=10)
    # net_1 = torchvision.models.convnext_base(num_classes=10)
    # net_1 = torchvision.models.resnet101(num_classes=10)
    # net_1 = torchvision.models.swin_b(num_classes=10)
    # net_1 = torchvision.models.vit_l_16(num_classes=10)
    # net_1 = timm.models.mambaout_base(num_classes=10)

# ##################################################AG-fashion-mnist%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ./checkpoint/ICPNet_fashion_MNIST_epoch33_bestacc_93.74_model.pth
    # ./checkpoint/vgg16_fashion_MNIST_epoch35_bestacc_94.12_model.pth
    # ./checkpoint/Resnet18_fashion_MNIST_epoch48_bestacc_94.02_model.pth
    # ./checkpoint/convnext_fashion_MNIST_epoch86_bestacc_93.33_model_nw=6.pth
    # ./checkpoint/Resnet101_fashion_MNIST_epoch55_bestacc_93.96_model.pth
    # ./checkpoint/swin_b_fashion_MNIST_epoch93_bestacc_90.62_model.pth
    # ./checkpoint/vit_l_16_fashion_MNIST_epoch92_bestacc_88.58_model.pth
    # ./checkpoint/mambaout_fashion_MNIST_epoch93_bestacc_92.45_model.pth

# ################################################################AG-MNIST%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ./checkpoint/icpnet_high_MNIST_bestacc_99.49_model.pth
    # ./checkpoint/vgg16_high_mnist_epoch41_bestacc_99.41_model.pth
    # ./checkpoint/ResNet18_high_MNIST_epoch35_bestacc_99.7_model.pth
    # ./checkpoint/convnext_high_mnist_epoch65_bestacc_99.45_model.pth
    # ./checkpoint/ResNet101_high_MNIST_epoch64_bestacc_99.72_model.pth
    # ./checkpoint/swin_b_high_MNIST_epoch77_bestacc_98.62_model.pth
    # ./checkpoint/ViT-l-16_high_MNIST_epoch84_bestacc_97.23_model.pth
    # ./checkpoint/mambaout_high_mnist_bestacc_99.37_model.pth

    weights = torch.load('./checkpoint/ICPNet_fashion_MNIST_epoch33_bestacc_93.74_model.pth', map_location='cpu')['model']
    net_1.load_state_dict(weights)
    print('transfer finishing')

    criterion = icpnet_model.edge_cls_loss(reduction='mean')

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    net_1.to(device)
    criterion.to(device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.1307],  # mnist_ag
        #                      std=[0.3081])
        transforms.Normalize(mean=[0.2910],  # ag-fashion-MNIST
                             std=[0.3126])
    ])

    ill_name_ = ['ag_i4_hor', 'ag_i4_ul', 'ag_i4_ur', 'ag_i4_ver', 'ag_i6_hor', 'ag_i6_ul', 'ag_i6_ur', 'ag_i6_ver',
                'ag_i8_hor', 'ag_i8_ul', 'ag_i8_ur', 'ag_i8_ver', 'ag_i10_hor', 'ag_i10_ul', 'ag_i10_ur', 'ag_i10_ver',
                'ag_i12_hor', 'ag_i12_ul', 'ag_i12_ur', 'ag_i12_ver', 'ag_i14_hor', 'ag_i14_ul', 'ag_i14_ur', 'ag_i14_ver']

    for i in range(len(ill_name_)):
        test_dataset = fashion_MNIST_AG(root=cfgs['dataset'], flag='test', transform=trans, ill_name=ill_name_[i])
        # test_dataset = AGMNIST_all(root=cfgs['dataset'], flag='test', transform=trans, ill_name=ill_name_[i])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=2)  # type: ignore

        inference(ill_name_[i])
