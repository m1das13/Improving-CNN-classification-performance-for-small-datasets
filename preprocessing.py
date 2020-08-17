import json
import os
import numpy as np
import PIL
import matplotlib.pyplot as plt
import math
import pickle
from torchvision import transforms
from PIL import ImageDraw
from skimage.color import rgb2gray
from torchvision.transforms.functional import affine
from scipy.ndimage import gaussian_filter

# ouptut image dimensions
Y, X = 40, 40

def resize_bounds(left, top, right, bottom): # maybe resize with padding of 10
    
    # calculate x and y dimensions
    yrange = bottom - top
    xrange = right - left
    
    # check to add range for x or y
    xyrange = [yrange, xrange]
    lessRange = xyrange.index(min(xyrange))
    
    # check how much range to add
    addRange = abs(yrange - xrange) / 2
    
    # add range accodingly
    left = (left - math.floor(addRange) if lessRange == 1 else left)
    top = (top - math.floor(addRange) if lessRange == 0 else top)
    right = (right + math.ceil(addRange) if lessRange == 1 else right)
    bottom = (bottom + math.ceil(addRange) if lessRange == 0 else bottom)
    return left, top, right, bottom

def randomly_zoom_out(left, top, right, bottom):
    padding = int(np.random.uniform(low=0.0, high=20.0))
    left -= padding
    top -= padding
    right += padding
    bottom += padding
    return left, top, right, bottom


def crop_img(img, scale=1.0):
    center_x, center_y = img.shape[1] / 2, img.shape[0] / 2
    width_scaled, height_scaled = img.shape[1] * scale, img.shape[0] * scale
    left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
    top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
    img_cropped = img[int(top_y):int(bottom_y), int(left_x):int(right_x)]
    return img_cropped

def add_gaussian_noise(rgb, ill_invar, mean=0, sigma=4):
    sigma = np.random.randint(sigma)
    rgb, ill_invar = np.array(rgb).astype(int), np.array(ill_invar).astype(int)
    # calculate Gaussian noise
    gaussian = np.random.normal(mean, sigma, (ill_invar.shape[0], ill_invar.shape[1], ill_invar.shape[2])).astype(int)

    rgb += gaussian[:,:,:3]
    ill_invar += gaussian 
    
    rgb[rgb < 0] = 0
    ill_invar[ill_invar < 0] = 0
    
    rgb[rgb > 255] = 255
    ill_invar[ill_invar > 255] = 255

    rgb = rgb.astype(np.uint8)
    ill_invar = ill_invar.astype(np.uint8)

    rgb = PIL.Image.fromarray(rgb)
    ill_invar = PIL.Image.fromarray(ill_invar)
    return rgb, ill_invar

def color_manipulate(rgb, ill_invar, b=(0.7,1.2), c=(0.7,1.3), s=(0.7,1.3), h=0):    
    trans_color = transforms.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)
    return trans_color(rgb), trans_color(ill_invar)

def blur_image(rgb, ill_invar, sigma=2):
    sigma = np.random.randint(sigma)
    
    rgb_arr = np.array(rgb)
    ill_invar_arr = np.array(ill_invar)
    
    rgb_arr = gaussian_filter(rgb_arr, sigma)
    ill_invar_arr = gaussian_filter(ill_invar_arr, sigma)
    
    return PIL.Image.fromarray(rgb_arr), PIL.Image.fromarray(ill_invar_arr)

def random_affine(img, ill_invar, left, top, right, bottom, angle=360, shear=0):
    angle = np.random.randint(0, 360)
    shear = np.random.randint(0, 30)
    # angle = np.random.choice([0,90,180,270,360])
    # shear = 0

    yrange = bottom - top
    xrange = right - left
    
    y_dim, x_dim = yrange , xrange 

    if yrange % 2 == 1:
        y_dim += 1
        x_dim += 1
        
    # for rgb image
    img_crop_doubleSize = img.crop((int(left - x_dim/2), 
                                    int(top - y_dim/2), 
                                    int(right + x_dim/2), 
                                    int(bottom + y_dim/2))) 
    
    rotated_img = affine(img_crop_doubleSize, angle, (0,0), 1, shear)
    img = crop_img(np.array(rotated_img), scale=.5)
    
    # for illumination invariant colorspace
    illInvar_crop_doubleSize = ill_invar.crop((int(left - x_dim/2), 
                                    int(top - y_dim/2), 
                                    int(right + x_dim/2), 
                                    int(bottom + y_dim/2))) 

    rotated_illInvar = affine(illInvar_crop_doubleSize, angle, (0,0), 1, shear)
    ill_invar = crop_img(np.array(rotated_illInvar), scale=.5)
    
    img = PIL.Image.fromarray(img)
    ill_invar = PIL.Image.fromarray(ill_invar)
    
    return img, ill_invar


# Input image directories and their labels
rgb_dir = 'xBD dataset/images/rgb'
ill_invar_dir = 'xBD dataset/images/ill_invar'
label_dir = 'xBD dataset/labels'

# Iutput directory with cropped images
output_dir = 'CNN_images/all'

# alphabetically sorted files in folder
sorted_images = sorted(os.listdir(rgb_dir))
sorted_labels = sorted(os.listdir(label_dir))

items = len(sorted_images)

# data augmentation gaussian parameters
mean = 0
sigma = 2

iterdict = {'no-damage': 1, 'minor-damage': 7, 'major-damage': 19, 'destroyed': 25}

unclass_count = 0
# loop over namesfiles in folder
for i, (image_file, label_file) in enumerate(zip(sorted_images, sorted_labels)):
    # check if filenames match
    if not image_file[:-4] == label_file[:-5]:
#         raise ValueError('Filenames do not match!')
        print(f'Filenames of file {i} do not match!')
        continue
        
    # load rbg image and corresponding illumination invariant image
    rgb_img = PIL.Image.open(f'{rgb_dir}/{image_file}')
    ill_invar = PIL.Image.open(f'{ill_invar_dir}/{image_file}')

    # convert rgb image to array for masking
    rgbArray = np.asarray(rgb_img)
    
    with open(f'{label_dir}/{label_file}') as label:

        # load json label info
        json_file = json.load(label)

        # iterate over all building polygons in image
        for j, building in enumerate(json_file['features']['xy']):

            try:
                damage_type = building['properties']['subtype']
            except:
                damage_type = 'no-damage'

            if damage_type == 'un-classified':
                unclass_count += 1
                continue
                
            
            # read polygon info and convert to boolean image 
            polygon = [tuple(np.asarray(tup.split(), dtype=np.float)) for tup in building['wkt'][10:-2].split(',')]
            maskIm = PIL.Image.new('1', (rgbArray.shape[1], rgbArray.shape[0]), 0)
            ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
            mask = np.array(maskIm)
            
            # crop image where mask value equals True
            y,x = np.where(mask == True)
            left, right = np.min(x), np.max(x)
            top, bottom = np.min(y), np.max(y)
            
            # resize bounds such that the resulting image is square
            left, top, right, bottom = resize_bounds(left, top, right, bottom)

            iterations = iterdict[damage_type]

            # data augmentation            
            for it in range(iterations):

                # randomly zoom out a bit
                nleft, ntop, nright, nbottom = randomly_zoom_out(left, top, right, bottom)
                rgb_cropped = rgb_img.crop((nleft, ntop, nright, nbottom))
                ill_invar_cropped = ill_invar.crop((nleft, ntop, nright, nbottom))

                rgb_cropped, ill_invar_cropped = random_affine(rgb_img, ill_invar, nleft, ntop, nright, nbottom)


                rgb_cropped, ill_invar_cropped = add_gaussian_noise(rgb_cropped, ill_invar_cropped, 
                                                                    mean, sigma)

                rgb_cropped, ill_invar_cropped = color_manipulate(rgb_cropped, ill_invar_cropped,
                                                                b=(0.7,1.2), c=(0.7,1.3), s=(0.7,1.3), h=0)

                rgb_cropped, ill_invar_cropped = blur_image(rgb_cropped, ill_invar_cropped, 
                                                          sigma=2)

                # resize images to specified size (Y, X) with LANCZOS filter
                if Y and X:
                    rgb_cropped = rgb_cropped.resize((Y, X), PIL.Image.LANCZOS)
                    ill_invar_cropped = ill_invar_cropped.resize((Y, X), PIL.Image.LANCZOS)

                # convert to array and bring in range [0,1]
                rgb_cropped = np.array(rgb_cropped) / 255
                ill_invar_cropped = rgb2gray(np.array(ill_invar_cropped)) # already in range [0,1]   

    #             print(rgb_cropped.shape)

                # save both cropped images in a 4 channel image
                cropped_4D = np.zeros((Y if Y else rgb_cropped.shape[0], X if X else rgb_cropped.shape[1], 4))
                cropped_4D[:, :, :3] = rgb_cropped
                cropped_4D[:, :, 3] = ill_invar_cropped

#                 save image in correct damage type directory
                output_path = f'{output_dir}/{damage_type}/{image_file[18:-13]}_{j:08d}_{it:02d}.npy'
                
                with open(output_path, 'wb') as f_out:
                    np.save(f_out, cropped_4D)
    #     break
    # break
    print(f'Done processing image {i+1}/{items}', end='\r')

print()
print('Done!', end='\r')