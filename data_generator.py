import tensorflow as tf
import numpy as np
import os
import cv2
from six import raise_from
import xml.etree.ElementTree as ET
import anchors
import warnings
import random

classes = {'body': 0}


class PascalVocGenerator(tf.keras.Sequential):
    def __init__(self, data_root_dir, image_dir, annots_dir, subset,
                 phi=0,
                 image_sizes=(512, 640, 768, 896, 1024, 1280, 1408),
                 batch_size=1,
                 shuffle_groups=True):
        self.data_root_dir = data_root_dir
        self.image_dir = image_dir
        self.annots_dir = annots_dir
        self.subset = subset
        self.names = [l.strip().split(None, 1)[0] for l in
                      open(os.path.join(self.data_root_dir, self.subset + '.txt')).readlines()]

        self.image_size = image_sizes[phi]
        self.anchor_parameters = anchors.AnchorParameters.default
        self.anchors = anchors.anchors_for_shape(
            (self.image_size, self.image_size), anchor_params=self.anchor_parameters)
        self.num_anchors = self.anchor_parameters.num_anchors()

        self.batch_size = batch_size
        self.groups = None
        self.group_images()

        self.classes = classes

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return 1  # len(self.classes)

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]


    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(self.data_root_dir, self.image_dir,
                            self.names[image_index] + '.jpg')
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _findNode(self, parent, name, parse=None):

        result = parent.find(name)
        if result is None:
            raise ValueError('missing element \'{}\''.format(name))
        if parse is not None:
            try:
                return parse(result.text)
            except ValueError as e:
                raise_from(ValueError(
                    'illegal value for \'{}\': {}'.format(name, e)), None)
        return result

    def __parse_annotation(self, element):
        """
        Parse an annotation given an XML element.
        """

        class_name = 'body'  # _findNode(element, 'name').text

        box = np.zeros((4,))
        label = self.name_to_label(class_name)

        bndbox = self._findNode(element, 'bndbox')
        box[0] = self._findNode(bndbox, 'xmin', parse=float) - 1
        box[1] = self._findNode(bndbox, 'ymin', parse=float) - 1
        box[2] = self._findNode(bndbox, 'xmax', parse=float) - 1
        box[3] = self._findNode(bndbox, 'ymax', parse=float) - 1

        return box, label

    def __parse_annotations(self, xml_root):
        """
        Parse all annotations under the xml_root.
        """
        annotations = {'labels': np.empty((0,), dtype=np.int32),
                       'bboxes': np.empty((0, 4))}
        for i, element in enumerate(xml_root.iter('object')):
            try:
                box, label = self.__parse_annotation(element)
            except ValueError as e:
                raise_from(ValueError(
                    'could not parse object #{}: {}'.format(i, e)), None)
            annotations['bboxes'] = np.concatenate(
                [annotations['bboxes'], [box]])
            annotations['labels'] = np.concatenate(
                [annotations['labels'], [label]])

        return annotations

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        filename = self.names[image_index] + '.xml'
        try:
            tree = ET.parse(os.path.join(
                self.data_root_dir, self.annots_dir, filename))
            return self.__parse_annotations(tree.getroot())
        except ET.ParseError as e:
            raise_from(ValueError(
                'invalid annotations file: {}: {}'.format(filename, e)), None)
        except ValueError as e:
            raise_from(ValueError(
                'invalid annotations file: {}: {}'.format(filename, e)), None)

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(
            image_index) for image_index in group]
        for annotations in annotations_group:
            assert (isinstance(annotations,
                               dict)), '\'load_annotations\' should return a list of dictionaries, received: {}'.format(
                type(annotations))
            assert (
                'labels' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'
            assert (
                'bboxes' in annotations), '\'load_annotations\' should return a list of dictionaries that contain \'labels\' and \'bboxes\'.'

        return annotations_group

    def filter_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] < 0) |
                (annotations['bboxes'][:, 1] < 0) |
                (annotations['bboxes'][:, 2] <= 0) |
                (annotations['bboxes'][:, 3] <= 0) |
                (annotations['bboxes'][:, 2] > image.shape[1]) |
                (annotations['bboxes'][:, 3] > image.shape[0])
            )[0]

            # delete invalid indices
            if len(invalid_indices):
                warnings.warn('Image with id {} (shape {}) contains the following invalid boxes: {}.'.format(
                    group[index],
                    image.shape,
                    annotations['bboxes'][invalid_indices, :]
                ))
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(
                        annotations[k], invalid_indices, axis=0)
            # if annotations['bboxes'].shape[0] == 0:
            #     warnings.warn('Image with id {} (shape {}) contains no valid boxes before transform'.format(
            #         group[index],
            #         image.shape,
            #     ))
        return image_group, annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def __len__(self):
        """
        Number of batches for generator.
        """
        return len(self.groups)

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_inputs_targets(group)
        return inputs, targets

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        order = list(range(self.size()))
        random.shuffle(order)

        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def preprocess_group_entry(self, image, annotations):
        """
        Preprocess image and its annotations.
        """
        # preprocess the image
        image, scale = self.preprocess_image(image)

        # apply resizing to annotations too
        annotations['bboxes'] *= scale
        return image, annotations

    def preprocess_image(self, image):
        # image, RGB
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            scale = self.image_size / image_height
            resized_height = self.image_size
            resized_width = int(image_width * scale)
        else:
            scale = self.image_size / image_width
            resized_height = int(image_height * scale)
            resized_width = self.image_size

        image = cv2.resize(image, (resized_width, resized_height))
        image = image.astype(np.float32)
        image /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image -= mean
        image /= std
        pad_h = self.image_size - resized_height
        pad_w = self.image_size - resized_width
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        return image, scale
    
    def preprocess_group(self, image_group, annotations_group):
        """
        Preprocess each image and its annotations in its group.
        """
        assert (len(image_group) == len(annotations_group))

        for index in range(len(image_group)):
            # preprocess a single group entry
            image_group[index], annotations_group[index] = self.preprocess_group_entry(image_group[index],
                                                                                       annotations_group[index])
        return image_group, annotations_group


    def clip_transformed_annotations(self, image_group, annotations_group, group):
        """
        Filter annotations by removing those that are outside of the image bounds or whose width/height < 0.
        """
        # test all annotations
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            # x1
            annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
            # y1
            annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
            # x2
            annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
            # y2
            annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            # test x2 < x1 | y2 < y1 | x1 < 0 | y1 < 0 | x2 <= 0 | y2 <= 0 | x2 >= image.shape[1] | y2 >= image.shape[0]
            small_indices = np.where(
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 3) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 3)
            )[0]

            # delete invalid indices
            if len(small_indices):
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)
                # import cv2
                # for invalid_index in small_indices:
                #     x1, y1, x2, y2 = annotations['bboxes'][invalid_index]
                #     label = annotations['labels'][invalid_index]
                #     class_name = self.labels[label]
                #     print('width: {}'.format(x2 - x1))
                #     print('height: {}'.format(y2 - y1))
                #     cv2.rectangle(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 255, 0), 2)
                #     cv2.putText(image, class_name, (int(round(x1)), int(round(y1))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
                # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                # cv2.imshow('image', image)
                # cv2.waitKey(0)
            filtered_image_group.append(image)
            filtered_annotations_group.append(annotations_group[index])

        return filtered_image_group, filtered_annotations_group

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        batch_images = np.array(image_group).astype(np.float32)
        return [batch_images]
    
    def compute_inputs_targets(self, group, debug=False):
        """
        Compute inputs and target outputs for the network.
        """
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # check validity of annotations
        # FilterDetections
        image_group, annotations_group = self.filter_annotations(
            image_group, annotations_group, group)

        # randomly apply visual effect
        # image_group, annotations_group = self.random_visual_effect_group(
        #     image_group, annotations_group)

        # randomly transform data
        # image_group, annotations_group = self.random_transform_group(image_group, annotations_group)

        # randomly apply misc effect
        # image_group, annotations_group = self.random_misc_group(
        #     image_group, annotations_group)

        # perform preprocessing steps
        image_group, annotations_group = self.preprocess_group(
            image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.clip_transformed_annotations(
            image_group, annotations_group, group)

        assert len(image_group) != 0
        assert len(image_group) == len(annotations_group)

        # compute network inputs
        inputs = self.compute_inputs(image_group, annotations_group)

        # compute network targets
        targets = self.compute_targets(image_group, annotations_group)

        if debug:
            return inputs, targets, annotations_group

        return inputs, targets

    def compute_targets(self, image_group, annotations_group):
        """
        Compute target outputs for the network using images and their annotations.
        """
        """
        Compute target outputs for the network using images and their annotations.
        """

        batches_targets = anchors.anchor_targets_bbox(
            self.anchors,
            image_group,
            annotations_group,
            num_classes=self.num_classes(),
        )
        return list(batches_targets)


if __name__ == '__main__':
    train_generator = PascalVocGenerator(
        'datasets/voc_trainval/VOC2007',
        'train',
        phi=0,
        skip_difficult=True,
        batch_size=1,
        misc_effect=None,
        visual_effect=None,
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    anchors = train_generator.anchors
    for batch_inputs, batch_targets in train_generator:
        image = batch_inputs[0][0]
        image[..., 0] *= std[0]
        image[..., 1] *= std[1]
        image[..., 2] *= std[2]
        image[..., 0] += mean[0]
        image[..., 1] += mean[1]
        image[..., 2] += mean[2]
        image *= 255.

        regression = batch_targets[0][0]
        valid_ids = np.where(regression[:, -1] == 1)[0]
        boxes = anchors[valid_ids]
        deltas = regression[valid_ids]
        class_ids = np.argmax(batch_targets[1][0][valid_ids], axis=-1)
        mean_ = [0, 0, 0, 0]
        std_ = [0.2, 0.2, 0.2, 0.2]

        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]

        x1 = boxes[:, 0] + (deltas[:, 0] * std_[0] + mean_[0]) * width
        y1 = boxes[:, 1] + (deltas[:, 1] * std_[1] + mean_[1]) * height
        x2 = boxes[:, 2] + (deltas[:, 2] * std_[2] + mean_[2]) * width
        y2 = boxes[:, 3] + (deltas[:, 3] * std_[3] + mean_[3]) * height
        for x1_, y1_, x2_, y2_, class_id in zip(x1, y1, x2, y2, class_ids):
            x1_, y1_, x2_, y2_ = int(x1_), int(y1_), int(x2_), int(y2_)
            cv2.rectangle(image, (x1_, y1_), (x2_, y2_), (0, 255, 0), 2)
            class_name = train_generator.labels[class_id]
            label = class_name
            ret, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
            cv2.rectangle(
                image, (x1_, y2_ - ret[1] - baseline), (x1_ + ret[0], y2_), (255, 255, 255), -1)
            cv2.putText(image, label, (x1_, y2_ - baseline),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.imshow('image', image.astype(np.uint8)[..., ::-1])
        cv2.waitKey(0)
        # 36864, 46080, 48384, 48960, 49104
        # if first_valid_id < 36864:
        #     stride = 8
        # elif 36864 <= first_valid_id < 46080:
        #     stride = 16
        # elif 46080 <= first_valid_id < 48384:
        #     stride = 32
        # elif 48384 <= first_valid_id < 48960:
        #     stride = 64
        # else:
        #     stride = 128
        pass
