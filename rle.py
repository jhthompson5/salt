import numpy as np
import matplotlib.pyplot as plt

def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return np.transpose(img.reshape(shape))

def rle_encode(img):
    #img = img.transpose()
    #new = [i for x in img for i in x]
    
    new = img
    #print(new[100:102])
    starts = []
    ends = []
    rle_mesh = []
    for i in range(len(new)):
        if i > 0:
            if new[i] == 1 and new[i-1] == 0:
                starts.append(i+1)
            if new[i] == 0 and new[i-1] == 1:
                ends.append(i+1)
        else:
            if new[i] == 1:
                starts.append(1)

    if(new[-1] == 1):
        ends.append(len(new)+1)
    for lo,hi in zip(starts,ends):
        rle_mesh.append(str(lo))
        rle_mesh.append(str(hi-lo))
    return " ".join(rle_mesh)


dat = ['5051 5151']
"""     '9 93 109 94 210 94 310 95 411 95 511 96 612 96 712 97 812 98 913 98 1015 97 1116 97 1216 98 1316 99 1416 8786',
    '48 54 149 54 251 53 353 52 455 51 557 50 659 49 762 47 864 46 966 45 1068 44 1171 42 1273 41 1376 39 1478 38 1581 36 1683 35 1785 34 1888 32 1990 31 2092 30 2195 28 2297 27 2399 26 2501 25 2602 25 2704 24 2806 23 2907 23 3009 22 3110 22 3212 21 3313 21 3414 21 3516 20 3617 20 3718 20 3819 20 3921 19 4022 19 4123 19 4225 18 4326 18 4428 17 4529 17 4631 16 4733 15 4834 15 4936 14 5038 13 5140 12 5242 11 5344 8',
    '1111 1 1212 1 1313 1 1414 1 1514 2 1615 2 1716 2 1817 2 1918 2 2018 3 2119 3 2220 3 2321 3 2422 3 2523 3 2624 3 2725 3 2826 3 2927 3 3028 3 3129 3 3230 3 3331 3 3432 3 3533 3 3636 1 3737 1 3838 1 3938 2 4039 2 4140 2 4240 3 4341 3 4442 3 4542 4 4643 4 4744 4 4844 5 4945 5 5046 5 5146 6 5247 6 5347 7 5448 7 5549 7 5649 8 5750 8 5851 8 5952 8 6053 8 6154 8 6255 8 6355 9 6456 9 6557 9 6659 8 6760 8 6861 8 6962 8 7063 8 7164 8 7265 8 7367 7 7468 7 7569 7 7670 7 7772 6 7873 6 7974 6 8075 6 8177 5 8278 5 8379 5 8480 5 8582 4 8683 4 8784 4 8885 4 8986 4 9087 4 9188 4 9289 4 9389 5 9490 5 9591 5 9692 5 9792 6 9893 6 9993 7 10094 7 10194 8',
    '1 1815 1819 90 1920 81 2021 73 2122 64 2223 55 2324 46 2425 36 2526 25 2627 13 2728 1'] """
for i in dat:
    img = rle_decode(i,(101,101))
    implt=plt.imshow(img,cmap='gray')
    plt.show() 
    print(rle_encode(i))
