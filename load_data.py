import sys
sys.path.append('./')

import xml.etree.ElementTree as ET
import cv2
import pickle
import os
import numpy as np

class pascal_voc(object):

    def __init__(self, phase, rebuild=False):
        self.cell_size = 7
        self.img_size = 448
        self.batch_size = 45
        self.VOC_PATH = r'C:\Users\user1\Desktop\Python\VOC2012'
        self.class_to_ind = {'aeroplane':0,
                           'bicycle': 1, 'bird':2,
                           'boat': 3, 'bottle':4,
                           'bus': 5, 'car':6,
                           'cat': 7, 'chair':8,
                           'cow': 9, 'diningtable':10,
                           'horse': 12, 'dog':11,
                           'motorbike': 13, 'person':14,
                           'pottedplant': 15, 'sheep':16,
                           'sofa': 17, 'train':18,
                           'tvmonitor': 19}

        self.phase = phase
        self.rebuild = rebuild

    def get(self):

        gt_labels = self._load()

        images = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25), dtype=np.float32)

        #random 変数作成
        idx = np.arange(0, len(gt_labels))
        np.random.shuffle(idx)
        idx = idx[:self.batch_size]

        count=0
        for i in idx:
            imname = gt_labels[i]['imname']
            images[count, :, :, :] = self._img_read(imname)
            labels[count, :, :, :] = gt_labels[i]['label']
            count +=1

        return images, labels

    def _img_read(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img / 255.0) * 2.0 - 1.0

        return img

    def _load(self):

        cache_file = os.path.join(os.path.dirname(__file__),'pascal_'+ self.phase + '_gt_labels.pkl')

        if os.path.exists(cache_file) and not self.rebuild:
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        if self.phase == 'train':
            txtname = os.path.join(self.VOC_PATH, 'ImageSets', 'Main', 'trainval.txt')
        if self.phase == 'test':
            txtname = os.path.join(self.VOC_PATH, 'ImageSets', 'Main', 'test.txt')
        with open(txtname, 'r') as f:
            self.img_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.img_index:
            label, num = self._load_annotation(index)

            if num == 0:
                continue

            imname = os.path.join(self.VOC_PATH, 'JPEGImages', index +'.jpg')

            gt_labels.append({'imname': imname,
                              'label':label})

        print('Saving gt_labels to: ' + cache_file)

        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)

        return gt_labels

    def _load_annotation(self, img_index):

        img_path = os.path.join(self.VOC_PATH, 'JPEGImages', img_index+'.jpg')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        w_ratio = self.img_size / img.shape[0] * 1.0
        h_ratio = self.img_size / img.shape[1] * 1.0

        label = np.zeros((self.cell_size, self.cell_size, 25))
        annot_file = os.path.join(self.VOC_PATH, 'Annotations', img_index+'.xml')
        tree = ET.parse(annot_file)

        for obj in tree.findall('object'):
            bbox = obj.find('bndbox')

            x1 = max(min((float((bbox.find('xmin').text)) - 1) * w_ratio, self.img_size - 1), 0)
            y1 = max(min((float((bbox.find('ymin').text)) - 1) * h_ratio, self.img_size -1) , 0)
            x2 = max(min((float((bbox.find('xmax').text)) - 1) * w_ratio, self.img_size - 1), 0)
            y2 = max(min((float((bbox.find('ymax').text)) - 1) * h_ratio, self.img_size -1) , 0)

            cls_ind = self.class_to_ind[obj.find('name').text]

            #[x_ctr, y_ctr, w, h]
            boxes = [(x1 + x2)/2.0,  (y1 + y2)/2.0, x2 - x1, y2 - y1]
            #x_ctrはどちらのBOXにあるかを研鑽
            x_ind = int(boxes[0] * self.cell_size/self.img_size)
            y_ind = int(boxes[1] * self.cell_size/self.img_size)

            #Confidence of [y_ind, x_ind]
            #
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1: 5] = boxes
            #物体の類別
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(tree.findall('object'))

images, labels = pascal_voc(phase='train', rebuild=False).get()
