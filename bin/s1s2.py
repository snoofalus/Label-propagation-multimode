
import os
import matplotlib.pyplot as plt
import hdf5storage
import numpy as np
#import h5py



work_dir = os.path.abspath('../workdir')
image_dir = os.path.abspath('../images/s1s2/by-image/')
s1s2_path = os.path.abspath(os.path.join(work_dir, 's1s2_new.mat'))


### hdf5storage
#s1s2_path = os.path.abspath('s1s2_new.mat')
s1s2_data = hdf5storage.loadmat(s1s2_path)['Attributes']
print('\n')
#print(type(s1s2_data))
#print(s1s2_data.shape) # 1458, 1830, 225
blue = s1s2_data[:, :, 15]
green = s1s2_data[:, :, 30]
red = s1s2_data[:, :, 45]
image = np.concatenate([red,green,blue], axis=2)
print(image.shape)

fig1 = plt.figure()
im = plt.imshow(image)

plt.show()

#print(s1s2_data[0:5, 0:5, 0:5])
#s1s2_data = hdf5storage.loadmat('s1s2_new.mat')['Attributes']
#print(len(s1s2_data.shape))


### h5py
'''
with h5py.File('s1s2_new.mat', 'r') as f:
    f.keys()
'''

### or
#I had a look at this issue: https://github.com/h5py/h5py/issues/726. If you saved your mat file with -v7.3 option, you should generate the list of keys with (under Python 3.x):
'''
with h5py.File('s1s2_new.mat', 'r') as file:
    print(list(file.keys()))
'''
#In order to access the variable a for instance, you have to use the same trick:

#with h5py.File('test.mat', 'r') as file:
#    a = list(file['a'])





