# Importing Image class from PIL module
from PIL import Image
import os

def crop(im):
    """
    input the image, return the cropped part of the image
    : im: image to crop
    : returns cropped image
    """
    # Size of the image in pixels (size of original image)
    width, height = im.size

    if width > height:
        diff = width - height
        top = 0
        bottom = height
        left = diff//2
        right = width - (diff//2)
        crop_img = im.crop((left, top, right, bottom))

    elif height > width:
        diff = height - width
        top = diff//2
        bottom = height - (diff//2)
        left = 0
        right = width
        crop_img = im.crop((left, top, right, bottom))
    else:
        crop_img = im
    
    return crop_img

def pad(im, color):
    """
    Pads the image to make it a square"
    : im: image to pad
    : color: color to pad the image
    : returns padded image
    """
    width, height = im.size

    # if widht and height are same, padding is not required.
    if width == height:
        pad_im = im
    
    elif width > height:
        pad_img = Image.new(im.mode, (width, width), color)
        pad_img.paste(im, (0, (width - height) // 2))
    
    else:
        pad_img = Image.new(im.mode, (height, height), color)
        pad_img.paste(im, ((height - width)//2, 0))
    
    return pad_img



# M1 data
path = "./M1_Database/"
image_names = os.listdir(path)
 
save_path = "./M1_Database_Cropped/"
if not os.path.exists('./M1_Database_Cropped'):
    os.makedirs('./M1_Database_Cropped')

count = 0

for image in image_names:
    im_path = path + image
    im_save_path = save_path + image
    

    # only use images that are not from myscope
    if image[-5:] != "m.jpg" and image[-4:] == ".jpg":
        im = Image.open(im_path)
        crop_img = crop(im)
        crop_img = crop_img.save(im_save_path)
        if count % 100 == 0:
            print(f"{count} Saved:{image}")
        count += 1
    
    # save myscope images without cropping
    if image[-5:] == "m.jpg":
        im = Image.open(im_path)
        im = pad(im, (256, 256, 256))
        im = im.save(im_save_path)
        if count % 100 == 0:
            print(f"{count} Saved:{image}")
        count += 1
        

# M2 data
path = "./M2_Database/"
image_names = os.listdir(path)

count = 0
 
save_path = "./M2_Database_Cropped/"
if not os.path.exists('./M2_Database_Cropped'):
    os.makedirs('./M2_Database_Cropped')

for image in image_names:
    im_path = path + image
    im_save_path = save_path + image
    im = Image.open(im_path)
    crop_img = crop(im)
    crop_img = crop_img.save(im_save_path)
    if count % 100 == 0:
       print("{count} Saved:{image}")
    count += 1

