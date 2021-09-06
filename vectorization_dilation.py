
import numpy as np

from PIL import Image
from PIL.ImageQt import ImageQt
import matplotlib.pyplot as plt
import timeit   

img = Image.open('./image/5.bmp')

def dilation_origin(img):
    img = np.array(img)
    (N,M) = img.shape
    print(img.shape)

    paddingImg = np.zeros((N+2, M+2), dtype=np.uint8)
    paddingImg[1:1+N, 1:1+M] = img.copy()
    dilate = np.zeros_like(img)


    for i in range(0,N,1):
        for j in range(0,M,1):
            temp = paddingImg[i:i+3, j:j+3]
            dilate[i][j] = np.max(temp)

    # plt.imshow(dilate, cmap = 'gray')
    # plt.show()



def dilation_line(img,N,M):


    paddingImg = np.zeros((N+2, M+2), dtype=np.uint8)
    paddingImg[1:1+N, 1:1+M] = img.copy()
    dilate = np.zeros((M,N), dtype=np.uint8)

    copy_array = np.zeros((3,N), dtype = np.uint8)
    copy_col = np.zeros((3,M+2), dtype = np.uint8)
    col_max = np.zeros((N,M+2),  dtype = np.uint8)

    for i in range(N):
        copy_col = paddingImg[i:i+3, :]
        col_max[i,:] = np.max(copy_col, axis=0)

    col_max = np.copy(col_max.transpose(), order='C')
    #dilate = np.copy(dilate.transpose(), order='C')
    #col_max = col_max.T
    #dilate = dilate.T

    for j in range(M):
        copy_array = col_max[j:j+3,:]
        dilate[j,:] = np.max(copy_array, axis=0)

    dilate = np.copy(dilate.transpose(), order='C')

    # copy_array = np.zeros((N+2,3), dtype=np.uint8)
    # copy_col = np.zeros((3, M), dtype=np.uint8)
    # row_max = np.zeros((N+2, M), dtype=np.uint8)

    # for i in range(M):
    #     copy_array = paddingImg[:,i:i+3]
    #     row_max[:,i]= np.max(copy_array, axis=1)

    # for j in range(N):
    #     copy_col = row_max[j:j+3, :]
    #     dilate[j,:] = np.max(copy_col, axis=0)


    

    # plt.imshow(dilate, cmap = 'gray')
    # plt.show()

    



def dilation(img):
    img = np.array(img)
    (N,M) = img.shape
    print(img.shape)

    # plt.subplot(1,2,1)
    # plt.imshow(img, cmap = 'gray')

    paddingImg = np.zeros((N+2, M+2), dtype=np.uint8)
    paddingImg[1:1+N, 1:1+M] = img.copy()
    dilate = np.zeros_like(img)

    row_max = np.zeros(N+2, dtype=np.uint8)

    for j in range(M):
        for i in range(N+2):
            max = paddingImg[i,j] if (paddingImg[i,j] > paddingImg[i,j+1]) else paddingImg[i,j+1]
            row_max[i] = paddingImg[i,j+2] if (paddingImg[i,j+2] > max) else max

            if i > 3 and i < N:
                max = row_max[i] if (row_max[i] > row_max[i+1]) else row_max[i+1]
                dilate[i,j] = row_max[i+2] if (row_max[i+2] > max) else max


    # plt.subplot(1,2,2)
    # plt.imshow(dilate, cmap = 'gray')
    # plt.show()


def dilation_move(img):
    img = np.array(img)
    (N,M) = img.shape
    print(img.shape) 

    paddingImg = np.zeros((N+2, M+2, 9), dtype=np.uint8)
    #paddingImg[1:1+N, 1:1+M,0] = img.copy()
    dilate = np.zeros_like(img)

    z = 0
    #이미지를 9방향으로 복사
    for i in range(3):
        for j in range(3):
            paddingImg[j:j+N, i:i+M,z] = img.copy()
            z = z+1

    max_array = np.zeros((N, M, 9), dtype=np.uint8)
    max_array = paddingImg[1:1+N, 1:1+M, :].copy()

    for j in range(M):
        for i in range(N):
            dilate[i,j] = max(max_array[i,j,:])


    
    # paddingImg = np.zeros((9, N+2, M+2), dtype=np.uint8)
    # dilate = np.zeros_like(img)
    # x = 0
    # for i in range(3):
    #     for j in range(3):
    #         paddingImg[x,j:j+N,i:i+M] = img.copy()
    #         x = x+1

    # max_array = np.zeros((9,N,M), dtype=np.uint8)
    # max_array = paddingImg[:, 1:1+N, 1:1+M].copy()

    # for j in range(M):
    #     for i in range(N):
    #         dilate[i,j] = max(max_array[:,i,j])

    # plt.imshow(dilate, cmap = 'gray')
    # plt.show()

img = np.array(img)
(N,M) = img.shape

start_time = timeit.default_timer()

dilation_line(img,N,M)
#dilation(img)
#dilation_origin(img)
#dilation_move(img)

terminate_time = timeit.default_timer()
print((terminate_time - start_time))

