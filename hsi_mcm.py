from PIL import Image
import tifffile as tf
import numpy as np
im=tf.imread('masks/000000_nir.tif')
imarray = np.array(im)
im1=tf.imread('masks/000000_rgb.tif')
imarray1 = np.array(im1)
print(imarray1.shape)


hsi_total = []
f_list = np.genfromtxt(('all.txt'), dtype='str')
print(f_list)

hsi_total=[]
for num in range(0,1):
    print(num)
    str_num = '0'+str(num)
    im=tf.imread('images/0000'+str_num+'_nir.tif')
    imarray = np.array(im)
    im_mask=tf.imread('masks/0000'+str_num+'_nir.tif')
    imarray_mask = np.array(im_mask)
    for i in range(len(imarray[0])):
        for j in range(len(imarray[0][0])):
            arr=[]
            for k in range(len(imarray)):
                arr=arr+[imarray[k,i,j]]
            if(im_mask[i][j]==0):
                arr=arr+[-1]
            else:
                arr=arr+[1]
            hsi_total=hsi_total+[arr]
            
print(hsi_total[0])

192*384
hsi_total=np.asarray(hsi_total)
#hsi_total=hsi_total[0:10000,:]

np.savetxt("hsi_4.csv", hsi_total, delimiter=",")

print(hsi_total.shape)

