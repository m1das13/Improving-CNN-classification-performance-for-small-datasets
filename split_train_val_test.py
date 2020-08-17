import os
import numpy as np

train_test_perc, train_val_perc = 0.8, 0.8

noDamage_dir = 'CNN_images/all/2/no-damage'
minDamage_dir = 'CNN_images/all/2/minor-damage'
majDamage_dir = 'CNN_images/all/2/major-damage'
destroyed_dir = 'CNN_images/all/2/destroyed'

train_dir = 'CNN_images/train/1'
test_dir = 'CNN_images/test/1'
val_dir = 'CNN_images/val/1'

# image shape
X, Y = 40,40

for i, directory in enumerate([noDamage_dir, minDamage_dir, majDamage_dir, destroyed_dir]):
    progress = 0
    listdir = os.listdir(directory)
    np.random.shuffle(listdir)
    split_train_test = round(len(listdir) * train_test_perc)
    split_train_val = round(split_train_test * train_val_perc)
    
    train_images, test_images = listdir[:split_train_test], listdir[split_train_test:]
    train_images, val_images = train_images[:split_train_val], train_images[split_train_val:]
    
    dirsize = len(listdir)

    for j, img_file in enumerate(train_images):
        with open(f'{directory}/{img_file}' , "rb") as img:
            image = np.load(img)
            if not (image.shape[0] == Y or image.shape[1] == X):
                print(f'{directory}/{img_file}')
                raise ValueError('Not of correct shape!')
            
            output_path = f'{train_dir}/{directory[17:]}/{img_file}'
            # print(output_path)
            # break
            with open(output_path, "wb") as f_out:
                np.save(f_out, image)
                
        progress += 1
        print(f'Processed training image {progress}/{dirsize} from {directory}', end='\r')
            
            
    for j, img_file in enumerate(val_images):
        with open(f'{directory}/{img_file}' , "rb") as img:
            image = np.load(img)

            if not (image.shape[0] == X or image.shape[1] == Y):
                print(f'{directory}/{img_file}')
                raise ValueError('Not of correct shape!')
                      
            output_path = f'{val_dir}/{directory[17:]}/{img_file}'
            # print(output_path)
            # break
            with open(output_path, "wb") as f_out:
                np.save(f_out, image)

        progress += 1
        print(f'Processed validation image {progress}/{dirsize} from {directory}', end='\r')

    for j, img_file in enumerate(test_images):
        with open(f'{directory}/{img_file}' , "rb") as img:
            image = np.load(img)

            if not (image.shape[0] == X or image.shape[1] == Y):
                print(f'{directory}/{img_file}')
                raise ValueError('Not of correct shape!')
                      
            output_path = f'{test_dir}/{directory[17:]}/{img_file}'
            # print(output_path)
            # break
            with open(output_path, "wb") as f_out:
                np.save(f_out, image)

        progress += 1
        print(f'Processed test image {progress}/{dirsize} from {directory}', end='\r')
    # break

print()
print("Done!")   