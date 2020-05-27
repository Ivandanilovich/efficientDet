import imgaug
import imgaug.augmenters as iaa

def get_base_aug_seq():
    return iaa.Sequential([
                            iaa.Sometimes(0.5, [iaa.Fliplr(0.5)]),
                            iaa.Sometimes(0.5, [iaa.Flipud(0.5)]),
                            iaa.Sometimes(0.1, [iaa.Grayscale(alpha=1.0)]),
                            iaa.Sometimes(0.5, [iaa.Rot90(k=2)]),
                            iaa.Sometimes(1, [iaa.Crop(px=(1, 80), keep_size=True)]),
                            iaa.Sometimes(0.1, [iaa.GaussianBlur(sigma=(0, 3.0))]),
                            iaa.Sometimes(0.9, [
                                iaa.Affine(
                                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}, 
                                    rotate=(-5, 5), 
                                    shear=(-2, 2),
                                    order=[0, 1], 
                                    cval=(0, 255), 
                                    mode=imgaug.ALL 
                                )
                            ]),
                            iaa.Sometimes(0.1, [iaa.AdditiveGaussianNoise(scale=(1, 0.05*255))]),
                            iaa.Sometimes(0.1, [iaa.Add((-40, 40))]),
                            iaa.Sometimes(0.1, [iaa.geometric.PerspectiveTransform(0.1)]),
                            iaa.Sometimes(0.7, [iaa.LinearContrast(alpha=(0.7,1.3))]),
                            iaa.Sometimes(0.05, [iaa.CLAHE(clip_limit=(1,5))]),
                            iaa.Sometimes(0.05, [iaa.ElasticTransformation(alpha=9, sigma=4)]),
                            iaa.Sometimes(0.05, [iaa.PiecewiseAffine(scale=(0,0.035))])
            ], random_order=True)


def get_flip_aug_seq():
    return iaa.Sequential([iaa.Fliplr(1.0)], random_order=True)