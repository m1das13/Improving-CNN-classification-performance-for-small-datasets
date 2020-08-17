from helpfiles.illumination_invariance import *
import pickle


# Add illumination invariant colorspace channel
directory = '../xBD dataset/images/rgb'
output_ill_invar = '../xBD dataset/images/ill_invar'

# output_4channel = '../xBD dataset/images/4channel'

# specify correct alpha for camera sensor (derived from peak spectral sensitivity)
alpha = 0.54
# alpha1 = 0.8

# alphabetically sorted files in folder
sorted_folder = sorted(os.listdir(directory))

# loop over namesfiles in folder
for i, filename in enumerate(sorted_folder):    
    # open image and calculate its shape
    img = skimread(f'{directory}/{filename}')
    imshape = img.shape
    
    # new empty four channel image
    img_4channel = np.zeros([imshape[0], imshape[1], 4], dtype=np.uint8)
    
    # calculate illumination invariant colorspace
    norm_ill_invar = illumination_invariant(img, alpha)
    
    # scale to values between [0-255]
    ill_invar = norm_ill_invar * 255
    
    # save image 
    plt.imsave(output_ill_invar + f'/{filename}', 
               np.asarray(ill_invar, dtype=np.uint8), cmap='gray')
    
    # insert color channels
    img_4channel[:, :, :3] = img # RGB
    img_4channel[:, :, 3] = ill_invar # 1D ill_invariant colorspace
    
    
    with open(output_ill_invar + f'/{filename}', "wb") as f_out:
        pickle.dump(ill_invar, f_out)

    # write 4 channel image to file
    # with open(output_4channel + f'/{filename}', "wb") as f_out:
    #     pickle.dump(img_4channel, f_out)
    
    # plot color and illumination invariant image
    # plt.figure(figsize=(15,15))
    # imshow_row([ (img, f"{filename[10:18]}{filename[23:31]}"), 
    #              (np.asarray(ill_invar, dtype=np.uint8), f"{filename[10:18]}{filename[23:31]}")])
    # plt.show()
    
    # if i == 3:
    #     break
