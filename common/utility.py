from urllib import request
import os


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
