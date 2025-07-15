from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


def read_from_pair_txt4linux(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]
    filenames = [c.split(' ') for c in filenames]
    filenames = [[os.path.join(path, c[0]),
                  os.path.join(path, c[1])] for c in filenames]
    return filenames


def read_from_mnist_txt4linux(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]
    filenames = [c.split(' ') for c in filenames]
    filenames = [[os.path.join(path, c[0]),
                  c[1]] for c in filenames]
    return filenames


class MNIST(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = read_from_mnist_txt4linux(root['MNIST'], 'train.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1] for im_name in filenames]

            self.edge_list = ['0' for _ in range(60000)]
            for i in range(len(self.im_list)):
                edge_path = self.im_list[i].split('/')
                edge_path[-2] = 'edge1'
                path_1 = edge_path[-1].split('.')
                edge_path[-1] = path_1[-2] + '.png'
                self.edge_list[i] = os.path.join('/', edge_path[1], edge_path[2], edge_path[3], edge_path[4], edge_path[5], edge_path[6])

        elif flag == 'test':
            filenames = read_from_pair_txt4linux(root['MNIST'], 'test.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1][-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        imagergb = Image.open(self.im_list[item], mode='r')

        if self.flag == 'train':
            cls_label = int(self.label_list[item])
            edge_label = np.array(Image.open(self.edge_list[item]).convert('L'))
            edge_label = Image.fromarray(edge_label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            cls_label = int(self.label_list[item])
            edge_label = Image.open(self.im_list[item])
        sampleName = ['images', 'cls_labels', 'edge_labels']
        sampleData = [imagergb, edge_label]

        if self.transform:
            sample, edge_label = self.transform(sampleData)
        else:
            sample = 0

        resample = {sampleName[0]: sample, sampleName[1]: cls_label, sampleName[2]: edge_label}

        return resample


class AGMNIST_all(Dataset):
    def __init__(self, root, flag='train', transform=None, ill_name='ag_i4_hor'):
        if flag == 'train':
            pass

        elif flag == 'test':
            filenames = read_from_mnist_txt4linux(root['high_AGMNIST_all'], 'test-224-ag.lst')
            for i in range(len(filenames)):
                path = filenames[i][0].split('/')
                path[-2] = ill_name
                filenames[i][0] = os.path.join(os.sep, *path)

            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1][-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        imagergb = Image.open(self.im_list[item], mode='r')  #（1, 224, 224）

        if self.flag == 'train':
            label = int(self.label_list[item])
        elif self.flag == 'test':
            label = int(self.label_list[item])
        sampleName = ['images', 'cls_labels']

        if self.transform:
            sample = self.transform(imagergb)
        else:
            sample = 0

        resample = {sampleName[0]: sample, sampleName[1]: label}
        return resample


class fashion_MNIST(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = read_from_mnist_txt4linux(root['fashion_MNIST'], 'train.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1] for im_name in filenames]

            self.edge_list = ['0' for _ in range(60000)]
            for i in range(len(self.im_list)):
                edge_path = self.im_list[i].split('/')
                edge_path[-2] = 'train_224_edge_nms'
                path_1 = edge_path[-1].split('.')
                edge_path[-1] = path_1[-2] + '.png'
                self.edge_list[i] = os.path.join('/', edge_path[1], edge_path[2], edge_path[3], edge_path[4], edge_path[5], edge_path[6])

        elif flag == 'test':
            filenames = read_from_pair_txt4linux(root['fashion_MNIST'], 'AGG_8916.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1][-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        imagergb = Image.open(self.im_list[item], mode='r')   #（1,28,28）

        if self.flag == 'train':
            cls_label = int(self.label_list[item])
            edge_label = np.array(Image.open(self.edge_list[item]).convert('L'))
            edge_label = Image.fromarray(edge_label.astype(np.float32) / 255.0)
        elif self.flag == 'test':
            cls_label = int(self.label_list[item])
            edge_label = Image.open(self.im_list[item])
        sampleName = ['images', 'cls_labels', 'edge_labels', 'img_ori']
        sampleData = [imagergb, edge_label]

        if self.transform:
            sample, edge_label = self.transform(sampleData)
        else:
            sample = 0

        resample = {sampleName[0]: sample, sampleName[1]: cls_label, sampleName[2]: edge_label}

        return resample


class fashion_MNIST_AG(Dataset):
    def __init__(self, root, flag='train', transform=None, ill_name='ag_i4_hor'):
        if flag == 'train':
            pass

        elif flag == 'test':
            filenames = read_from_mnist_txt4linux(root['fanshion_mnist_AG'], 'AG_test_8916.lst')
            for i in range(len(filenames)):
                path = filenames[i][0].split('/')
                path[-2] = ill_name
                filenames[i][0] = os.path.join(os.sep, *path)

            self.im_list = [im_name[0] for im_name in filenames]
            self.label_list = [im_name[1][-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        imagergb = Image.open(self.im_list[item], mode='r')

        if self.flag == 'train':
            pass

        elif self.flag == 'test':
            label = int(self.label_list[item])
        sampleName = ['images', 'cls_labels']

        if self.transform:
            sample = self.transform(imagergb)
        else:
            sample = 0

        resample = {sampleName[0]: sample, sampleName[1]: label}
        return resample
