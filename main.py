import numpy as np
import cv2 as cv

im1 = cv.imread("im1.png")
im2 = cv.imread("im2.png")

#Isolate the area of the image containing the colour scale
#Did this twice just in case both images don't always use 
#exactly the same colours
colour_scale1 = im1[125:765, 80:150, :]
colour_scale2 = im2[125:765, 80:150, :]

#Remove duplicate colours
colours1 = np.unique(colour_scale1.reshape(-1, colour_scale1.shape[-1]), axis=0)
colours2 = np.unique(colour_scale2.reshape(-1, colour_scale2.shape[-1]), axis=0)

#The line below always returns 32, as it shoud from counting the colours
#on the scale manually
unique_colours = len(colours1)

vals = np.linspace(-1, 1.2, unique_colours)

#Just for debugging
def showim(im):
    cv.imshow("the image", im)
    key = cv.waitKey(0)
    if key == 27:
        cv.destroyWindow("the image")

#Just so you can show two images at once and they will fit on the screen
def hstack_images(image1, image2):
    image1 = np.array(image1)
    image2 = np.array(image2)
    image1 = cv.resize(image1, (0, 0), None, 0.25, 0.25)
    image2 = cv.resize(image2, (0, 0), None, 0.25, 0.25)
    stacked = np.hstack((image1, image2))
    return stacked

def resize(im):
    im = cv.resize(im, (0, 0), None, 0.5, 0.5)
    return im


#Test for colour 1

#Remove the left half of the image, it just contains the scale
#which we don't need anymore
im1 = im1[:,960:,:]
im2 = im2[:,960:,:] 

#Setup final masks
final_matrix1 = np.ndarray((1080, 960))
final_matrix1[:,:] = 0
final_matrix2 = np.ndarray((1080, 960))
final_matrix2[:,:] = 0

for i in range(0, unique_colours):
    #The cv.inRange function takes an input image, followed by two colours
    #and returns a matrix composed of 1s and 0s, 1 indicating that the pixel
    #at that index falls within the two colours
    #Colours in image don't perfectly match the colours in scale so just multiply by 1.02
    #to increase the range slightly
    mask1 = cv.inRange(im1, colours1[i] * 0.98, colours1[i] * 1.02)
    print("found ", len(np.nonzero(mask1)[0]), " pixels")
    #Edit the final matrix for image one, put the value in at all pixel
    #locations of that colour
    final_matrix1[mask1 != 0] = vals[i]
    mask2 = cv.inRange(im2, colours2[i] * 0.98, colours2[i] * 1.02)
    final_matrix2[mask2 != 0] = vals[i]
    #showim(hstack_images(mask1, mask2))

#Here we should calculate they delta, just for testing i'm just using the final
#matrix of image one
value_delta_matrix = final_matrix1
#Delta matrix will be constructed below to contain colours
delta_matrix = np.ndarray((1080, 960, 3))


for i, v in enumerate(vals):
    #This should output an image composed of only the colours in the array colours1
    #There should be NO white pixels
    delta_matrix[value_delta_matrix == v] = colours1[i]

showim(resize(delta_matrix))
   

   