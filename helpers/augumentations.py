import os
import albumentations as A
import cv2



# Shape : Albumentations transform
transform_shape = A.Compose([
    A.SomeOf([
        A.ImageCompression(quality_lower=30, quality_upper=55, p=0.3),
        A.MotionBlur(blur_limit=7, p=0.3),
    ], n=2),
    A.GridDistortion(num_steps=3, distort_limit=0.25, p=0.2),
    A.RandomCrop(width=200, height=180, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                       p=0.7, border_mode=cv2.BORDER_CONSTANT),
    A.ElasticTransform(alpha_affine=9, p=0.2, border_mode=cv2.BORDER_CONSTANT),
    A.OpticalDistortion(distort_limit=0.2, p=0.2,
                        border_mode=cv2.BORDER_CONSTANT),
    A.ZoomBlur(max_factor=1.1, p=0.2),
    A.Resize(width=320, height=280, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.3))

# Color : Albumentations transform
transform_color = A.Compose([
    # Quality : Jpeg compression, multiplicative noise, downscale
    A.OneOf([
        A.RandomBrightnessContrast(p=0.3),
        A.Equalize(p=0.3),
        A.ImageCompression(quality_lower=30, quality_upper=55, p=0.3),
        A.MultiplicativeNoise(p=0.2),
        A.Downscale(scale_min=0.4, scale_max=0.6, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.2, 0.8), p=0.1),
        A.PixelDropout(dropout_prob=0.1, p=0.1),
        A.Spatter(intensity=0.3,p=0.1),
        A.Superpixels(p=0.1),
        A.GlassBlur(sigma=0.1, 
                    max_delta=2, 
                    iterations=1, 
                    p=0.1),
    ]),
    # Weather : Dropouts, rain, snow, sun flare, fog
    A.OneOf([
        A.RandomRain(drop_length=4, 
                     blur_value=4, 
                     p=0.1),
        A.RandomSnow(p=0.1, brightness_coeff=1),
        A.RandomSunFlare(src_radius=100, 
                         num_flare_circles_lower=2,
                         num_flare_circles_upper=4, 
                         p=0.1),
        A.RandomFog(fog_coef_lower=0.1, 
                    fog_coef_upper=0.5,
                    alpha_coef=0.5, 
                    p=0.1),
    ]),
    A.Resize(width=320, height=280, always_apply=True),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))

# All : Full transform
transform_all = A.Compose([
    A.SomeOf([transform_color], n=3, p=0.5),
    A.SomeOf([transform_shape], n=3, p=0.5),
], bbox_params=A.BboxParams(format='yolo', min_area=100, min_visibility=0.2))


def Augment(imagePath: str,
            outputName : str,
            outputDirectory: str,
            transformations) -> str:
    ''' Read image, augment image and bboxes and save it to new file. '''

    # Read image
    image = cv2.imread(imagePath)

    # Augmentate image
    transformed = transformations(image=image, bboxes=[])

    # Create filename
    outputFilepath = os.path.join(outputDirectory, outputName)

    # Image : Save
    cv2.imwrite(outputFilepath, transformed['image'])

    return outputFilepath
