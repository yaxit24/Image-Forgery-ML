import numpy as np
from  skimage import color
from PIL import Image, ExifTags
# pil stands for the pyhton image library
# here the skimage is Scikit-Image (image processing library)

def load_image_pil(path_or_file):
    pil = Image.open(path_or_file).convert('RGB') # here we are opening the image here and then converting to format of red, green, Blue.

    # *Image is used to 
          #  1) OPEN Image.
          #  2) Convert Formats
          #  3) Resize and Rotate
          #  4) Process Pixels
    arr = np.array(pil) # convert the Image(3D) array to  Numpy array.
    return arr, pil

def get_Y_channel(RGB_img_array):
    # RGB_img_array : HxWx3(format), uint8(dtype) where the it the output of the above function here.

    # COLOR used for conversions: (the below is the diff format of the iamges)
        #RGB → Grayscale
        # RGB → YCbCr
        # RGB → HSV  .. etc.
    
    ycbcr = color.rgb2ycbcr(RGB_img_array) # here it is conveted to the 'float64' 
    #Converted  RGB → YCbCr([R, G, B]  →  [Y, Cb, Cr]) where the Y  = Luminance (Brightness), Cb = Blue             difference, Cr = Red difference
    Y = [..., 0] # here we are just extracting the luminance(brightness)
    # here the the just bright ness we extracted here is used for the  1) Edge detection,
    # 2) Imagecomparison, 3) Compression, 4) ML preprocessing
    return Y, ycbcr

    
    