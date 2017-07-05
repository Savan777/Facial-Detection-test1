import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

src = cv2.imread("exercise-1.jpg")
img = Image.open("exercise-1.jpg")
res = src.copy()
img = img.convert('YCbCr')
ycbcr = np.array(img)

Y = 0
Cb = 1
Cr = 2

YCbCr = list(img.getdata())
imYCbCr = np.reshape(YCbCr, (img.size[1], img.size[0], 3))
imYCbCr = imYCbCr.astype(np.uint8)

yChannel = Image.fromarray(imYCbCr[:,:,Y], "L") #L is for 8-bit pixels (mode)
cbChannel = Image.fromarray(imYCbCr[:,:,Cb], "L")
crChannel = Image.fromarray(imYCbCr[:,:,Cr], "L")

white = [255,255,255]
black = [0,0,0]

for (x,y), pixel in np.ndenumerate(cbChannel):
    if( pixel >=105 and pixel <= 135 ):
        res[x,y,:] = white
    else:
        res[x,y,:] = black

for (x,y), pixel in np.ndenumerate(crChannel):
    if( pixel >=140 and pixel <= 165 ):
        res[x,y,:] = white
    else:
        res[x,y,:] = black



plt.imshow(res)
plt.show()


#Show images with effect of each channel
# yChannel.show()
# cbChannel.show()
# crChannel.show()










