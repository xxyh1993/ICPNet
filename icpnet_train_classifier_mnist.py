import torch
import torch.utils
import torch.utils.data
import yaml
# import torchvision.transforms as transforms
import transforms
# import torchvision
from data_edge2 import MNIST, AGMNIST_all
import icpnet_model
import time
import os
from datetime import datetime
import random_seed
# from visdom import Visdom
# from visual_loss_edge2 import Visualizer
# import torch.nn as nn


def AG_test(illname):
    # inference stage
    length = len(test_AG_MNIST_dataloader)
    total_imgs = test_AG_MNIST_dataset.__len__()
    t_time = 0.
    correct_top1 = 0
    correct_top5 = 0
    correct = 0
    loss_test = 0.
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_AG_MNIST_dataloader, start=0):

            b, _, h, w = data['images'].shape
            x_2 = torch.zeros((b, 128, h//4, w//4), device='cuda:3')
            x_3 = torch.zeros((b, 256, h//8, w//8), device='cuda:3')
            x_4 = torch.zeros((b, 512, h//16, w//16), device='cuda:3')
            states = [x_2, x_3, x_4]

            images = data['images'].to(device)
            labels = data['cls_labels'].to(device)
            start_time = time.time()

            prediction, edge_pred = net(images, states)
            loss_test += criterion(prediction, labels, flag='test')
            duration = time.time() - start_time
            t_time += duration

            _, pred_cls_top1 = torch.max(prediction.data, 1)
            # bs >= 1
            correct_classes = pred_cls_top1.eq(labels.data).cpu().sum()
            correct_top1 += correct_classes
            for i in range(b):
                _, pred_cls_top5 = torch.sort(prediction.data, 1, True)
                pred_cls_top5_ = pred_cls_top5[i][0:5]
                labels_ = labels[i].expand_as(pred_cls_top5_)
                correct = pred_cls_top5_.eq(labels_.data).cpu().sum()
                correct_top5 += correct

            # print_epoch = 10
            # visualization by visdom
            # if i % print_epoch == print_epoch - 1:
            #     vis_img.image(img_ori[0], win='AG_image' + illname, opts={'title': 'AG_image' + illname})
            #     vis_img.image(edge_pred[0], win='AG_edge' + illname, opts={'title': 'AG_edge' + illname})

        acc_top1 = correct_top1 / total_imgs * 100.
        acc_top5 = correct_top5 / total_imgs * 100.
        format_str = '%s, epoch: %d, ill_name: %s, acc_TOP1: %.4f, acc_TOP5: %.4f, loss_test: %.3f, avg_time: %.3f, avg_FPS:%.3f'
        log_ag.write(format_str % (datetime.now(), epoch, illname, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time) + '\n')
        print(format_str % (datetime.now(), epoch, illname, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time))

        # plot top1 curve by visdomn
        # AG_image_test_top1.append(acc_top1)
        # if illname == 'ag_i4_ver':
        #     vis_loss.plot_i4(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()
        # elif illname == 'ag_i6_ver':
        #     vis_loss.plot_i6(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()
        # elif illname == 'ag_i8_ver':
        #     vis_loss.plot_i8(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()
        # elif illname == 'ag_i10_ver':
        #     vis_loss.plot_i10(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()
        # elif illname == 'ag_i12_ver':
        #     vis_loss.plot_i12(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()
        # elif illname == 'ag_i14_ver':
        #     vis_loss.plot_i14(AG_image_test_top1, epoch+1)
        #     AG_image_test_top1.clear()


def inference():
    # inference stage
    length = len(test_dataloader)
    total_imgs = test_dataset.__len__()
    t_time = 0.
    correct_top1 = 0
    correct_top5 = 0
    correct = 0
    loss_test = 0.
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, start=0):
            if i % 100 == 99:
                log.write('process %3d/%3d images' % (i+1, length) + '\n')
                print('process %3d/%3d images' % (i+1, length))
            b, _, h, w = data['images'].shape

            x_2 = torch.zeros((b, 128, h//4, w//4), device='cuda:3')
            x_3 = torch.zeros((b, 256, h//8, w//8), device='cuda:3')
            x_4 = torch.zeros((b, 512, h//16, w//16), device='cuda:3')
            states = [x_2, x_3, x_4]

            images = data['images'].to(device)
            cls_labels = data['cls_labels'].to(device)
            start_time = time.time()

            prediction, edge_pred = net(images, states)
            duration = time.time() - start_time
            t_time += duration
            loss_test += criterion(prediction, cls_labels, flag='test')

            _, pred_cls_top1 = torch.max(prediction.data, 1)
            # bs >= 1
            correct_classes = pred_cls_top1.eq(cls_labels.data).cpu().sum()
            correct_top1 += correct_classes
            for i in range(b):
                _, pred_cls_top5 = torch.sort(prediction.data, 1, True)
                pred_cls_top5_ = pred_cls_top5[i][0:5]
                labels_ = cls_labels[i].expand_as(pred_cls_top5_)
                correct = pred_cls_top5_.eq(labels_.data).cpu().sum()
                correct_top5 += correct

            # print_epoch = 100
            # visualization by visdom
            # if i % print_epoch == print_epoch - 1:
            #     vis_img.image(edge_pred[0], win='test_edge_pred', opts={'title': 'test_edge_pred'})
            #     vis_img.image(img_ori[0], win='image_ori', opts={'title': 'image_ori'})

        acc_top1 = correct_top1 / total_imgs * 100.
        acc_top5 = correct_top5 / total_imgs * 100.
        format_str = '%s, epoch: %d, acc_TOP1: %.4f, acc_TOP5: %.4f, loss_test: %.3f, avg_time: %.6f, avg_FPS:%.3f'
        log.write(format_str % (datetime.now(), epoch, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time) + '\n')
        print(format_str % (datetime.now(), epoch, acc_top1, acc_top5, loss_test/length, t_time/length, length/t_time))

    return acc_top1, loss_test/length


if __name__ == '__main__':
    # visdom setting
    # vis_loss = Visualizer(env='train_icpnet')
    # vis_img = Visdom(env='img_visual_icpnet')

    with open('./cfgs_edge2.yaml', 'r', encoding='UTF-8') as f:
        cfgs = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists('./icpnet'):
        os.makedirs('./icpnet')
    name = cfgs['name']['MNIST']
    log = open('./icpnet/'+'1train_' + name + '#########.txt', 'a+')
    name_ag = cfgs['name']['high_AGMNIST']
    log_ag = open('./icpnet/'+'1test_illu_' + name_ag + '#########.txt', 'a+')

    # random seed
    random_seed.setup_seed(3407)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    print('==> Preparing data..')
    trans = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[[0.1307], [0.]], std=[[0.3081], [1.]])
    ])

    test_trans = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[[0.1307], [0.1307]],std=[[0.3081], [0.3081]])])

    AG_MNIST_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])])

    # MNIST data
    train_dataset = MNIST(root=cfgs['dataset'], flag='train', transform=trans)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfgs['batch_size'], shuffle=True, pin_memory=True, num_workers=6, persistent_workers=True)
    test_dataset = MNIST(root=cfgs['dataset'], flag='test', transform=test_trans)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=30, shuffle=False, num_workers=2)


    # AG_MNIST data
    ill_name = ['ag_i4_hor', 'ag_i4_ul', 'ag_i4_ur', 'ag_i4_ver', 'ag_i6_hor', 'ag_i6_ul', 'ag_i6_ur', 'ag_i6_ver',
                'ag_i8_hor', 'ag_i8_ul', 'ag_i8_ur', 'ag_i8_ver', 'ag_i10_hor', 'ag_i10_ul', 'ag_i10_ur', 'ag_i10_ver',
                'ag_i12_hor', 'ag_i12_ul', 'ag_i12_ur', 'ag_i12_ver', 'ag_i14_hor', 'ag_i14_ul', 'ag_i14_ur', 'ag_i14_ver']

    # print('==> Building model..')
    net = icpnet_model.ICPNet(num_classes=10)

    criterion = icpnet_model.edge_cls_loss(reduction='mean')

    # optimizer
    if cfgs['method'] == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=cfgs['lr'],
                                     weight_decay=cfgs['weight_decay'],
                                     betas=(0.9, 0.999))
    elif cfgs['method'] == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(),
                                      lr=cfgs['lr'],
                                      weight_decay=cfgs['weight_decay'],
                                      betas=(0.9, 0.999))
    elif cfgs['method'] == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=cfgs['lr'],
                                    momentum=cfgs['momentum'],
                                    weight_decay=cfgs['weight_decay'])

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgs['max_epoch'], eta_min=5.0e-5, last_epoch=-1)

    net.to(device)
    criterion.to(device)

    pre_training = 'False'
    if pre_training == 'True':
        net.load_state_dict(torch.load('./checkpoint/1MNIST_ex86_epoch25_bestacc_99.49_model.pth')['model'])

    # resuming from checkpoint
    start_epoch = -1
    checkpoint_index = 0
    if os.path.exists('./checkpoint/' + name + '_last_model.pth') and torch.load('./checkpoint/' + name + '_last_model.pth')['epoch'] != (cfgs['max_epoch']-1):
        interrupt = torch.load('./checkpoint/' + name + '_last_model.pth')
        net.load_state_dict(interrupt['model'])
        optimizer.load_state_dict(interrupt['optimizer'])
        start_epoch = interrupt['epoch']
        checkpoint_index = 1
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfgs['max_epoch'], eta_min=5.0e-5, last_epoch=start_epoch)
        log.write('Training has resumed' + '\n')
        print('Training has resumed')
    elif os.path.exists('./checkpoint/' + name + '_last_model.pth') and torch.load('./checkpoint/' + name + '_last_model.pth')['epoch'] == (cfgs['max_epoch']-1):
        log.write('Finishing training' + '\n')
        print('Finishing training')

    best_acc = 0.
    iterations = 0
    train_acc = []
    AG_image_test_top1 = []

    # train
    for epoch in range(start_epoch + 1, cfgs['max_epoch']):
        net.train()

        # # ##plot cosine lr_curve
        # log.write(cfgs['method'] + '\t' + str(scheduler.get_last_lr()[0]) + '\n')
        # print(cfgs['method'], '\t', scheduler.get_last_lr()[0])
        # # vis_loss.plot_lr(scheduler.get_last_lr()[0], epoch)  # drawing the learning rate curve by visdom

        # customized learning rate rules
        if checkpoint_index == 1:
            checkpoint_index += 1
            icpnet_model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'])
            log.write(cfgs['method'] + '\t' + str(optimizer.state_dict()['param_groups'][0]['lr']) + '\n')
            print(cfgs['method'], '\t', optimizer.state_dict()['param_groups'][0]['lr'])
            # vis_loss.plot_lr(optimizer.state_dict()['param_groups'][0]['lr'], epoch)  # drawing the learning rate curve by visdom
        else:
            icpnet_model.learning_rate_decay(optimizer, epoch, decay_rate=cfgs['decay_rate'])
            log.write(cfgs['method'] + '\t' + str(optimizer.state_dict()['param_groups'][0]['lr']) + '\n')
            print(cfgs['method'], '\t', optimizer.state_dict()['param_groups'][0]['lr'])
            # vis_loss.plot_lr(optimizer.state_dict()['param_groups'][0]['lr'], epoch)  # drawing the learning rate curve by visdom

        running_loss = 0.0
        cls_loss_total = 0.0
        edge_loss_total = 0.0

        for i, data in enumerate(train_dataloader, start=0):
            optimizer.zero_grad()
            b, _, h, w = data['images'].shape
            x_2 = torch.zeros((b, 128, h//4, w//4), device='cuda:3')
            x_3 = torch.zeros((b, 256, h//8, w//8), device='cuda:3')
            x_4 = torch.zeros((b, 512, h//16, w//16), device='cuda:3')
            states = [x_2, x_3, x_4]

            images = data['images'].to(device)
            cls_labels = data['cls_labels'].to(device)
            edge_labels = data['edge_labels'].to(device)

            start_time = time.time()

            prediction, edge_pred = net(images, states)

            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            duration = time.time() - start_time

            loss, cls_loss, edge_loss = criterion(prediction, cls_labels, edge_pred, edge_labels)
            loss.backward()
            optimizer.step()

            # iterations += 1
            # drawing loss curve by visdom
            # if i % 20 == 19:
            #     vis_loss.plot_loss(cls_loss.item(), iterations)  # type: ignore
            #     vis_loss.plot_edge_loss(edge_loss.item(), iterations)  # type: ignore

            print_epoch = 200
            # image visualization by visdom
            # if i % print_epoch == print_epoch - 1:
            #     vis_img.image(img_ori[0], win='image_ori', opts={'title': 'image_ori'})
            #     vis_img.image(edge_pred[0], win='edge_pred', opts={'title': 'edge_pred'})
            #     vis_img.image(edge_labels[0] * 255, win='edge_GT', opts={'title': 'edge_GT'})

            running_loss += loss.item()
            cls_loss_total += cls_loss.item()
            edge_loss_total += edge_loss.item()
            if i % print_epoch == print_epoch - 1:
                _, pred_classes = torch.max(prediction.data, 1)
                total_imgs = prediction.shape[0]  # shape: [b, num_classes]
                correct_classes = pred_classes.eq(cls_labels.data).cpu().sum()
                acc = 100.*(correct_classes/total_imgs)
                train_acc.append(acc.item())
                examples_per_sec = cfgs['batch_size'] / duration
                sec_per_batch = float(duration)
                format_str = '%s: step [%d, %5d/%4d], acc=%.4f, loss = %.3f, cls_loss = %.3f, edge_loss = %.3f (%.1f examples/sec; %.3f sec/batch)'
                log.write(format_str % (datetime.now(), epoch, i + 1, len(train_dataloader), acc, running_loss/print_epoch, cls_loss_total/print_epoch,
                                        edge_loss_total/print_epoch, examples_per_sec, sec_per_batch) + '\n')
                print(format_str % (datetime.now(), epoch, i + 1, len(train_dataloader), acc, running_loss/print_epoch, cls_loss_total/print_epoch,
                                    edge_loss_total/print_epoch, examples_per_sec, sec_per_batch))
                running_loss = 0.
                cls_loss_total = 0.
                edge_loss_total = 0.

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # inference stage
        acc_top1, test_loss = inference()

        # plot acc@top1 curve
        # vis_loss.plot_acc_stack([np.mean(train_acc), acc_top1.item()], iterations)  # type: ignore
        # train_acc.clear()

        # # ##########test && plot AG_MNIST@top1 curve
        for i in range(len(ill_name)):
            test_AG_MNIST_dataset = AGMNIST_all(root=cfgs['dataset'], flag='test', transform=AG_MNIST_trans, ill_name=ill_name[i])
            test_AG_MNIST_dataloader = torch.utils.data.DataLoader(test_AG_MNIST_dataset, batch_size=20, shuffle=False, num_workers=2)
            AG_test(ill_name[i])

        # update lr
        # scheduler.step()

        # save model
        state = {'model': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'epoch': epoch,
                 'acc': acc_top1}
        torch.save(state, './checkpoint/' + name + '_last_' + cfgs['save_name'])
        if acc_top1 > best_acc:
            best_acc = acc_top1
            torch.save(state, './checkpoint/' + '1' + name + '_epoch' + str(epoch) + '_bestacc_' + str(round(float(best_acc), 4)) + '_' + cfgs['save_name'])

    log.write('Finished Training' + '\n')
    print('Finished Training')
