#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
from skimage.io import imread as skimread
import numpy as np
import matplotlib.pyplot as plt


# # 1. Calculate illumination invariant colorspace

# In[ ]:


def illumination_invariant(image, alpha):
    # replace zeros with nan values to prevent log(0)
    image = np.where(image==0, np.nan, image)

    # take the log of rgb values
    log_r = np.log(image[:,:,0])
    log_g = np.log(image[:,:,1])
    log_b = np.log(image[:,:,2])
    
    # replace nan values with zero
    log_r = np.nan_to_num(log_r)
    log_g = np.nan_to_num(log_g)
    log_b = np.nan_to_num(log_b)

    # illumination invariant colorspace calculation
    ill_invar = np.array(0.5 + log_g - alpha * log_b - (1 - alpha) * log_r)
    return ill_invar

def imshow_row(imttllist, axs=False):
    n = len(imttllist)
    for i, imttl in enumerate(imttllist):
        if imttl is None:
            continue
        im, ttl = imttl
        plt.subplot(1,n,i+1)
        if len(im.shape) == 2:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)
        if not axs:
            plt.axis('off')
        plt.title(ttl)


# # 2. Test on satellite images

# In[ ]:


# directory = '../../xBD dataset/images/rgb'
# # imagefile = '../../xBD dataset/images/rgb/hurricane-michael_00000012_post_disaster.png'

# alpha = 0.54

# # alphabetically sorted files in folder
# sorted_folder = sorted(os.listdir(directory))

# # loop over namesfiles in folder
# for i, filename in enumerate(sorted_folder):
#     # open image
#     img = skimread(f'{directory}/{filename}')
    
#     # calculate illumination invariant colorspace
#     ill_invar = illumination_invariant(img, alpha)
#     plot_ill_invar = np.asarray(ill_invar * 255, dtype=np.uint8)
    
#     # plot color and illumination invariant image
#     plt.figure(figsize=(15,15))
#     imshow_row([ (img, f"{filename[10:18]}{filename[23:31]}"), 
#                  (plot_ill_invar, f"{filename[10:18]}{filename[23:31]}")])
#     plt.show()
    
#     if i == 3:
#         break


# # 3. Test on my own images

# In[ ]:


# directory = '../../test_images'
# # imagefile = '../../xBD dataset/images/rgb/hurricane-michael_00000012_post_disaster.png'

# alpha = 0.3

# # alphabetically sorted files in folder
# sorted_folder = sorted(os.listdir(directory))

# # loop over namesfiles in folder
# for i, filename in enumerate(sorted_folder):
#     # open image
#     img = skimread(f'{directory}/{filename}')
    
#     # calculate illumination invariant colorspace
#     ill_invar = illumination_invariant(img, alpha)
#     plot_ill_invar = np.asarray(ill_invar * 255, dtype=np.uint8)
    
#     # plot color and illumination invariant image
#     plt.figure(figsize=(15,15))
#     imshow_row([ (img, f"{filename[10:18]}{filename[23:31]}"), 
#                  (plot_ill_invar, f"{filename[10:18]}{filename[23:31]}")])
#     plt.show()
#     if i == 1:
#         break


# In[ ]:




