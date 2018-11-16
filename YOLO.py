import sys
sys.path.append('./')

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

class YOLONET(object):

    def __init__(self, is_training=True):
        self.cell_size = 7
        self.img_size = 448
        self.batch_size = 45
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

        self.num_class = len(self.class_to_ind)
        self.num_bbox = 2
        self.num_output = (self.cell_size * self.cell_size * (self.num_bbox*5 + self.num_class))
        self.class_scale = 2.0
        self.boundary1 = self.cell_size*self.cell_size*self.num_class
        self.boundary2 = self.cell_size*self.cell_size*(self.num_class + 2)
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.num_bbox),
            (self.num_bbox, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32, shape=[self.batch_size, self.img_size, self.img_size, 3])
        self.logits = self._buildnet(self.images, self.num_output, is_training, keep_prob=0.5)

        if is_training:
            self.labels = tf.placeholder(tf.float32, shape=[self.batch_size, self.cell_size, self.cell_size, 25])
            self.loss = self._loss_layer(self.logits, self.labels)

    def _buildnet(self,
                  images,
                  num_output,
                  is_training,
                  keep_prob=0.5,
                  scope='yolo_swift'):

        with tf.variable_scope(scope) as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn = tf.nn.leaky_relu,
                                weights_regularizer=slim.l2_regularizer(0.0005),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

                net = tf.pad(images, [[0, 0], [3, 3], [3, 3], [0, 0]], name='pad_1')
                net = slim.conv2d(net, 64, [7, 7], 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, [3, 3], scope='conv_4')
                net = slim.max_pool2d(net, [2, 2], scope='pool_5')
                net = slim.conv2d(net, 128, [1, 1], scope='conv_6')
                net = slim.conv2d(net, 256, [3, 3], scope='conv_7')
                net = slim.conv2d(net, 256, [1, 1], scope='conv_8')
                net = slim.conv2d(net, 512, [3, 3], scope='conv_9')
                net = slim.max_pool2d(net, [2, 2], scope='pool_10')
                net = slim.conv2d(net, 256, [1, 1], scope='conv_11')
                net = slim.conv2d(net, 512, [3, 3], scope='conv_12')
                net = slim.conv2d(net, 256, [1, 1], scope='conv_13')
                net = slim.conv2d(net, 512, [3, 3], scope='conv_14')
                net = slim.conv2d(net, 256, [1, 1], scope='conv_15')
                net = slim.conv2d(net, 512, [3, 3], scope='conv_16')
                net = slim.conv2d(net, 256, [1, 1], scope='conv_17')
                net = slim.conv2d(net, 512, [3, 3], scope='conv_18')
                net = slim.conv2d(net, 512, [1, 1], scope='conv_19')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv_20')
                net = slim.max_pool2d(net, [2, 2], scope='pool_21')
                net = slim.conv2d(net, 512, [1, 1], scope='conv_22')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv_23')
                net = slim.conv2d(net, 512, [1, 1], scope='conv_24')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv_25')
                net = slim.conv2d(net, 1024, [3, 3],  scope='conv_26')
                net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]], name='pad_27')
                net = slim.conv2d(net, 1024, [3, 3], 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv_29')
                net = slim.conv2d(net, 1024, [3, 3], scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 4096, scope='fc_33')
                net = slim.dropout(net, keep_prob, is_training=is_training, scope='drop_34')
                net = slim.fully_connected(net, num_output, activation_fn=None,scope='fc_35') #[None, 1470]

                return net

    def _loss_layer(self, predicts, labels):


        predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size,self.cell_size, self.cell_size, 20])

        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                    [self.batch_size,self.cell_size, self.cell_size, self.num_bbox])

        predict_boxes = tf.reshape(predicts[:,  self.boundary2:self.num_output],
                                    [self.batch_size,self.cell_size, self.cell_size, self.num_bbox, 4])


        response = tf.reshape(labels[:, :, :, 0], [self.batch_size,self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size,self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.num_bbox, 1] )
        classes = labels[:, :, :, 5: ]


        offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.num_bbox])

        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
        offset_tran = tf.transpose(offset, (0, 2, 1, 3))

        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size, #x_ctr
                                       (predict_boxes[:, :, :, :, 1] + offset_tran) /self.cell_size, #y_ctr
                                       tf.square(predict_boxes[:, :, :, :, 2]), #width)
                                       tf.square(predict_boxes[:, :, :, :, 3])], axis=-1) #height


        iou_predict_truth = self._cal_iou(predict_boxes_tran, boxes)
        print(iou_predict_truth)
        object_mask = tf.reduce_max(iou_predict_truth, 3, keepdims=True)
        # the return of (iou_predict_truth >= object_mask) is ture and flase
        # Use tf.cast change tf.bool into tf.float32, if True = 1.0, if False = 0.0
        # if use first predict bbox to predict it will be [1, 0], if use second predict box it will be [0, 1]
        # with nothing in predict box is going to be [0, 0]
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask # is 1 when there is no object in the cell i, else 0.

        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset, #x_ctr
                               boxes[:, :, :, :, 1] * self.cell_size - offset, #y_ctr
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])], axis=-1)

        object_mask_box = tf.expand_dims(object_mask, 4)
        boxes_delta = object_mask_box * (predict_boxes - boxes_tran)
        boxes_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='box_loss') * 5 #(5 is coord_score)

        object_delta = object_mask*(predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='ojbect_loss')

        no_object_delta = no_object_mask*predict_scales
        no_object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(no_object_delta), axis=[1, 2, 3]), name='no_ojbect_loss') * 0.5

        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')


    def _cal_iou(self, boxes1, boxes2, scope='iou'):

        with tf.variable_scope(scope) as scope:
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                 axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                 axis=-1)


            left_top = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2]) #(X1, Y1)
            right_bottom = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:]) #(X2, Y2)

            intersection = tf.maximum(0.0, right_bottom -left_top)

            intersection_area = intersection[..., 0] * intersection[..., 1]

            boxes1_area = boxes1[..., 2] * boxes1[..., 3]
            boxes2_area = boxes2[..., 2] * boxes2[..., 3]

            unique_area = boxes1_area + boxes2_area - intersection_area

            return tf.clip_by_value(intersection_area / unique_area, 0.0, 1.0)


YOLONET(is_training=True).loss
