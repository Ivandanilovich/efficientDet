from efficientnet import EfficientNetB0
import tensorflow as tf
import os
from data_generator import PascalVocGenerator
from matplotlib import pyplot as plt
import numpy as np
import cv2
from model import efficientdet
from losses import smooth_l1, focal

if __name__=='__main__':
    # print('here')
    # # efn = EfficientNetB0()
    # # print(efn)
    # p = tf.io.gfile.glob('C:/Users/ivand/Desktop/dataset/annots/*.xml')
    # print(len(p))
    # train_generator = PascalVocGenerator(
    #     data_root_dir='C:/Users/ivand/Desktop/dataset/',
    #     image_dir='images',
    #     annots_dir='annots',
    #     subset='train',
    #     phi=0,
    #     batch_size=4
    # )
    # d= 4

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # anchors = train_generator.anchors
    # for batch_inputs, batch_targets in train_generator:
    #     image = batch_inputs[0][0]
    #     image[..., 0] *= std[0]
    #     image[..., 1] *= std[1]
    #     image[..., 2] *= std[2]
    #     image[..., 0] += mean[0]
    #     image[..., 1] += mean[1]
    #     image[..., 2] += mean[2]
    #     image *= 255.
    #     # plt.imshow(np.array(image,dtype='int'))
    #     # plt.show()

    #     regression = batch_targets[0][0]
    #     valid_ids = np.where(regression[:, -1] == 1)[0]
    #     boxes = anchors[valid_ids]
    #     deltas = regression[valid_ids]
    #     class_ids = np.argmax(batch_targets[1][0][valid_ids], axis=-1)
    #     mean_ = [0, 0, 0, 0]
    #     std_ = [0.2, 0.2, 0.2, 0.2]

    #     width = boxes[:, 2] - boxes[:, 0]
    #     height = boxes[:, 3] - boxes[:, 1]

    #     x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
    #     y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
    #     x2 = boxes[:, 2] + (deltas[:, 0] * std_[2] + mean_[2]) * width # TODO: old generator
    #     y2 = boxes[:, 3] + (deltas[:, 0] * std_[3] + mean_[3]) * height
    #     for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
    #         x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
    #         cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
    #         class_name = 'body' # = train_generator.labels[class_id]
    #         label = class_name
    #         ret, baseline = cv2.getTextSize(
    #             label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    #         cv2.rectangle(
    #             image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
    #         cv2.putText(image, label, (x1_, y2_ - baseline),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    #     cv2.imshow('image', image.astype(np.uint8)[..., ::-1])
    #     cv2.waitKey(0)



    #     break
    # print('here')
    # edf = efficientdet()
    # print(edf.summary())

    model, prediction_model = efficientdet(0, num_classes=1,
                                           num_anchors=9)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss={
        'regression': smooth_l1(),
        'classification': focal()})

    d=4


