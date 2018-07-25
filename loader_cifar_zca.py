from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import torch
import random

class CIFAR10(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
       
    #data_file = 'cifar10_zca/cifar10_gcn_zca_v2.npz' 
    nclass = 10
    split_list = ['label', 'unlabel', 'valid', 'test']

    def __init__(self, root, split='train',
                 transform=None, target_transform=None,
                 download=False, boundary=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        assert(boundary<10)
        print("Boundary: ", boundary)
        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        # load data
        self.data = np.load(root)
        #self.data = np.load(self.data_file)
        #self.train_data_zca = self.data['train_x'].transpose(0,3,1,2)
        #self.train_labels_zca = self.data['train_y']
        #self.test_data_zca = self.data['test_x'].transpose(0,3,1,2)
        #self.test_labels_zca = self.data['test_y']

        # now load the picked numpy arrays
        if self.split is 'label' or self.split is 'unlabel' or self.split is 'valid':
            
            self.train_data = self.data['train_x'].astype(np.float32).transpose(0,3,1,2)
            #self.train_data = np.concatenate(self.train_data)
            self.train_labels = self.data['train_y'].astype(int)
            print(self.train_data.shape)
            print(self.train_labels.shape)
            if boundary is not 0:
                bidx = 5000 * boundary 
                self.train_data = [self.train_data[bidx:],self.train_data[:bidx]]
                self.train_data = np.concatenate(self.train_data)
                self.train_labels = [self.train_labels[bidx:],self.train_labels[:bidx]]
                self.train_labels = np.concatenate(self.train_labels)

            train_datau = []
            train_labelsu = []
            train_data1 = []
            train_labels1 = []
            valid_data1 = []
            valid_labels1 = []
            num_labels_valid = [0 for _ in range(self.nclass)]
            num_labels_train = [0 for _ in range(self.nclass)]
            for i in range(self.train_data.shape[0]):
                tmp_label = self.train_labels[i]
                if num_labels_valid[tmp_label] < 500:
                    valid_data1.append(self.train_data[i])
                    valid_labels1.append(self.train_labels[i])
                    num_labels_valid[tmp_label] += 1
                elif num_labels_train[tmp_label] < 400:
                    train_data1.append(self.train_data[i])
                    train_labels1.append(self.train_labels[i])
                    num_labels_train[tmp_label] += 1

                    #train_datau.append(self.train_data[i])
                    #train_labelsu.append(self.train_labels[i])
                else:
                    train_datau.append(self.train_data[i])
                    train_labelsu.append(self.train_labels[i])

            if self.split is 'label':
                self.train_data = train_data1
                self.train_labels = train_labels1

                self.train_data = np.concatenate(self.train_data)
                self.train_data = self.train_data.reshape((len(train_data1), 3, 32, 32))
                self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr = self.train_data.shape[0]
                #print(self.train_data1[:1,:1,:5,:5])
                #print(self.train_labels1[:10])
                #print(self.train_data[:1,:1,:5,:5])
                #print(self.train_labels[:10])
                print('Label: ',num_tr) #label
            
                #self.midx=0
                #self.idx_offset = num_tr_ul - (num_tr_ul//num_tr) * num_tr
                #print('Offset: :',self.idx_offset)

            elif self.split is 'unlabel':
                self.train_data_ul = train_datau
                self.train_labels_ul = train_labelsu

                self.train_data_ul = np.concatenate(self.train_data_ul)
                self.train_data_ul = self.train_data_ul.reshape((len(train_datau), 3, 32, 32))
                self.train_data_ul = self.train_data_ul.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr_ul = self.train_data_ul.shape[0]
                print('Unlabel: ',num_tr_ul) #unlabel

            elif self.split is 'valid':
                self.valid_data = valid_data1
                self.valid_labels = valid_labels1
 
                self.valid_data = np.concatenate(self.valid_data)
                self.valid_data = self.valid_data.reshape((len(valid_data1), 3, 32, 32))
                self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  # convert to HWC
                
                num_val = self.valid_data.shape[0]
                print('Valid: ',num_val) #valid
                #print(self.valid_data[:1,:1,:5,:5])
                #print(self.valid_labels[:10])
            
        elif self.split is 'test':
            #self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            #self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_data = self.data['test_x'].astype(np.float32) 
            self.test_labels = self.data['test_y'].astype(int) 

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split is 'label':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split is 'unlabel':
            img, target = self.train_data_ul[index], self.train_labels_ul[index]
        elif self.split is 'valid':
            img, target = self.valid_data[index], self.valid_labels[index]
        elif self.split is 'test':
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img)
        img1 = np.copy(img)
        #img1 = Image.fromarray(img1)
        if self.split is 'label' or self.split is 'unlabel':
            img = random_crop(img, 32, padding=2)
            img = horizontal_flip(img, 0.5)
            img = img.copy()
            img = torch.from_numpy(img)
            img = img + torch.randn_like(img) * 0.15
            img = img.permute(2,0,1)
            #img = self.transform(img)

            img1 = random_crop(img1, 32, padding=2)
            img1 = horizontal_flip(img1, 0.5)
            img1 = img1.copy()
            img1 = torch.from_numpy(img1)
            img1 = img1 + torch.randn_like(img1) * 0.15
            img1 = img1.permute(2,0,1)
            #img1 = self.transform(img1)
        else:
            img = torch.from_numpy(img)
            img = img.permute(2,0,1)

            img1 = torch.from_numpy(img1)
            img1 = img1.permute(2,0,1)
 
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img1

    def __len__(self):
        if self.split is 'label':
            return len(self.train_data)
        elif self.split is 'unlabel':
            return len(self.train_data_ul)
        elif self.split is 'valid':
            return len(self.valid_data)
        elif self.split is 'test':
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def horizontal_flip(image, rate=0.5):
    if random.random() < rate:
        #image = np.flip(image,1).copy()
        image = image[:, ::-1, :]
    return image

def random_crop(image, crop_size, padding=4):
    crop_size = check_size(crop_size)
    image = np.pad(image,((padding,padding),(padding,padding),(0,0)),'constant',constant_values=0)
    h, w, _ = image.shape
    top = random.randrange(0, h - crop_size[0])
    left = random.randrange(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size

if __name__ == '__main__':

    '''
    for i in range(10):
        print("Boundary %d///////////////////////////////////////"%i)
        data_train = CIFAR10('/tmp', split='label', download=True, transform=None, boundary=i)
        data_train_ul = CIFAR10('/tmp', split='unlabel', download=True, transform=None, boundary=i)
        data_valid = CIFAR10('/tmp', split='valid', download=True, transform=None, boundary=i)
        data_test = CIFAR10('/tmp', split='test', download=True, transform=None, boundary=i)

        print("Number of data")
        print(len(data_train))
        print(len(data_train_ul))
        print(len(data_valid))
        print(len(data_test))
    '''

    import torch.utils.data as data
    from math import ceil
    
    batch_size = 230
    
    labelset = CIFAR10('/tmp', split='label', download=True, transform=None, boundary=0)
    unlabelset = CIFAR10('/tmp', split='unlabel', download=True, transform=None, boundary=0)

    for i in range(90,256):
        batch_size = i
        label_size = len(labelset)
        unlabel_size = len(unlabelset)
        iter_per_epoch = int(ceil(float(label_size + unlabel_size)/batch_size))
        batch_size_label = int(ceil(float(label_size) / iter_per_epoch))
        batch_size_unlabel = int(ceil(float(unlabel_size) / iter_per_epoch))
        iter_label = int(ceil(float(label_size)/batch_size_label))
        iter_unlabel = int(ceil(float(unlabel_size)/batch_size_unlabel))
        if iter_label == iter_unlabel:
            print('Batch size: ', batch_size)
            print('Iter/epoch: ', iter_per_epoch)
            print('Batch size (label): ', batch_size_label)
            print('Batch size (unlabel): ', batch_size_unlabel)
            print('Iter/epoch (label): ', iter_label)
            print('Iter/epoch (unlabel): ', iter_unlabel)
    

    label_loader = data.DataLoader(labelset, batch_size=batch_size_label, shuffle=True)
    label_iter = iter(label_loader)

    unlabel_loader = data.DataLoader(unlabelset, batch_size=batch_size_unlabel, shuffle=True)
    unlabel_iter = iter(unlabel_loader)
    
    print(len(label_iter))
    print(len(unlabel_iter))
    



