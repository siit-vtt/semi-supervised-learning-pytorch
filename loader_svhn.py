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


class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'label': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'unlabel': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'valid': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='label',
                 transform=None, target_transform=None, download=False, boundary=0):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]
        assert(boundary<10)
        print('Boundary: ', boundary)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.train_data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.train_labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.train_labels, self.train_labels == 10, 0)
        self.train_data = np.transpose(self.train_data, (3, 2, 0, 1))

        if self.split is 'label' or self.split is 'unlabel' or self.split is 'valid':
            if boundary is not 0:
                bidx = 7000 * boundary
                self.train_data = [self.train_data[bidx:], self.train_data[:bidx]]
                self.train_data = np.concatenate(self.train_data)
                self.train_labels = [self.train_labels[bidx:], self.train_labels[:bidx]]
                self.train_labels = np.concatenate(self.train_labels)
 
            print(self.split)
            train_datau = []
            train_labelsu = []
            train_data1 = []
            train_labels1 = []
            valid_data1 = []
            valid_labels1 = []
            num_labels_train = [0 for _ in range(10)]
            num_labels_valid = [0 for _ in range(10)]
            
            for i in range(self.train_data.shape[0]):
                tmp_label = self.train_labels[i]
                if num_labels_valid[tmp_label] < 732:
                    valid_data1.append(self.train_data[i])
                    valid_labels1.append(self.train_labels[i])
                    num_labels_valid[tmp_label] += 1
                elif num_labels_train[tmp_label] < 100:
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
                #self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr = self.train_data.shape[0]
                print('Label: ',num_tr) #label
 
            elif self.split is 'unlabel':
                self.train_data_ul = train_datau
                self.train_labels_ul = train_labelsu

                self.train_data_ul = np.concatenate(self.train_data_ul)
                self.train_data_ul = self.train_data_ul.reshape((len(train_datau), 3, 32, 32))
                #self.train_data_ul = self.train_data_ul.transpose((0, 2, 3, 1))  # convert to HWC

                num_tr_ul = self.train_data_ul.shape[0]
                print('Unlabel: ',num_tr_ul) #unlabel

            elif self.split is 'valid':
                self.valid_data = valid_data1
                self.valid_labels = valid_labels1
 
                self.valid_data = np.concatenate(self.valid_data)
                self.valid_data = self.valid_data.reshape((len(valid_data1), 3, 32, 32))
                #self.valid_data = self.valid_data.transpose((0, 2, 3, 1))  # convert to HWC
                
                num_val = self.valid_data.shape[0]
                print('Valid: ',num_val) #valid
                #print(self.valid_data[:1,:1,:5,:5])
                #print(self.valid_labels[:10])
    
        else:
            print(self.split)
            self.test_data = self.train_data.reshape((len(self.train_data), 3, 32, 32))
            #self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            self.test_labels = self.train_labels

	    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.split is 'label':
            img, target = self.train_data[index], int(self.train_labels[index])
        elif self.split is 'unlabel':
            img, target = self.train_data_ul[index], int(self.train_labels_ul[index])
        elif self.split is 'valid':
            img, target = self.valid_data[index], int(self.valid_labels[index])
        elif self.split is 'test':
            img, target = self.test_data[index], int(self.test_labels[index])
	            

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img1 = np.copy(img)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img1 = Image.fromarray(np.transpose(img1, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
            img1 = self.transform(img1)

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
        else:
            assert(False)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


if __name__ == '__main__':

    ''' 
    for i in range(10):
        print("Boundary %d///////////////////////////////////////"%i)
        data_train = SVHN('/tmp', split='label', download=True, transform=None, boundary=i)
        data_train_ul = SVHN('/tmp', split='unlabel', download=True, transform=None, boundary=i)
        data_valid = SVHN('/tmp', split='valid', download=True, transform=None, boundary=i)
        data_test = SVHN('/tmp', split='test', download=True, transform=None, boundary=i)

        print("Number of data")
        print(len(data_train))
        print(len(data_train_ul))
        print(len(data_valid))
        print(len(data_test))
    
    '''
    import torch.utils.data as data
    from math import ceil
    
    batch_size = 230
    
    labelset = SVHN('/tmp', split='label', download=True, transform=None, boundary=0)
    unlabelset = SVHN('/tmp', split='unlabel', download=True, transform=None, boundary=0)

    for i in range(100,256):
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
    
    
