import os

def get_photo_bucket_id(photo_url):
    temp = os.path.split(photo_url)
    temp = os.path.split(temp[0])
    return temp[1]

def list_to_dict(input_list):
    return {k:v for v,k in enumerate(input_list)}
