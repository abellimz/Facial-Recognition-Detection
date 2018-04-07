from urllib2 import Request as request

import coremltools
import os

import cv2

from common.constants import EXTENSION_JPG


def get_photo_bucket_id(photo_url):
    temp = os.path.split(photo_url)
    temp = os.path.split(temp[0])
    return temp[1]

def photo_url_to_image_path(image_dir, photo_url):
    """ Converts photo url to image path, downloads image if not available"""
    bucket_id = get_photo_bucket_id(photo_url)
    bucket_path = os.path.join(image_dir, bucket_id)
    image_basename = os.path.basename(photo_url)
    image_filename = os.path.join(bucket_path, image_basename)
    if not os.path.exists(bucket_path):
        os.makedirs(bucket_path)
    if not os.path.exists(image_filename):
        response = request.urlopen(photo_url)
        data = response.read()
        with open(image_filename, "wb") as f:
            f.write(data)
    return image_filename

def list_to_dict(input_list):
    return {k:v for v,k in enumerate(input_list)}

def make_cropped_images(image_path, crop_image_dir,
                        crop_basename, crop_rects,
                        save_crops):
    """ Crops a given image path and returns a list of file paths
        to the cropped images in the same order as crop_rects.
    """
    crop_filenames = []
    for idx, (x, y, w, h) in enumerate(crop_rects):
        crop_filename = os.path.join(
            crop_image_dir, crop_basename + '_' + str(idx) + EXTENSION_JPG)
        if save_crops:
            if not os.path.exists(os.path.dirname(crop_filename)):
                os.makedirs(os.path.dirname(crop_filename))
            img = cv2.imread(image_path)
            crop_img = img[y:y + h, x:x + w]
            cv2.imwrite(crop_filename, crop_img)
        crop_filenames.append(crop_filename)
    return crop_filenames

def save_coreml_keras(model, model_path, class_labels=None,
                      input_names=None, image_input_names=None, output_names=None,
                      image_scale=1.0, red_bias=0.0, green_bias=0.0,
                      blue_bias=0.0):
    coreml_model = coremltools.converters.keras \
        .convert(model, input_names, output_names,
                 image_input_names=image_input_names,
                 class_labels=class_labels,
                 image_scale=image_scale,
                 red_bias=red_bias,
                 green_bias=green_bias,
                 blue_bias=blue_bias)
    coreml_model.save(model_path)