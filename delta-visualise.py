import numpy as np
import cv2 as cv
import os
import argparse


"""
WARNING: Do NOT run this more than once in iPython Kernel, for some
reason you get slightly different results every time, which makes no sense
I will fix it later, but I'm not really sure why it happens.
Just navigate to directory and "python main.py"
Or just straight up run it
"""

"""
Future work: Also detect unique colours from actual image then compare
to colours found in the colour scale and match them, this will result in
perfect reconstruction without the little black lines everywhere
"""

#USER DEFINED
min_delta_val = -0.5
max_delta_val = 0.5
#Delta values outside of the ranges specified above will be shown as the max delta colour.

#Just for debugging
############################################################
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

#Pass this the delta matrix constructed from only final matrix one
#Use to compare to image one, they should be the same
#Any diffferences are due to multiplying the colours by 1.02
#Fiddle with those coefficients to find closest match
#Ideally the whole system should just reconstruct the image
def compare(delta):
    im = cv.imread("im1.png")
    im = im[:,960:,:] / 255
    stacked = hstack_images(im, delta)
    #showim(stacked)
############################################################

#In this function you should calculate the delta as you see fit
def calculate_delta(matrix1, matrix2):
    #Just leaving this empty for now
    ##FOR EXAMPLE:
    delta = matrix2 - matrix1
    return delta

def make_delta(im1, im2):
    #Isolate the area of the image containing the colour scale
    #Did this twice just in case both images don't always use 
    #exactly the same colours
    colour_scale1 = im1[125:765, 80:150, :]
    colour_scale2 = im2[125:765, 80:150, :]

    #Remove duplicate colours
    colours1 = np.unique(colour_scale1.reshape(-1, colour_scale1.shape[-1]), axis=0)
    colours2 = np.unique(colour_scale2.reshape(-1, colour_scale2.shape[-1]), axis=0)

    #The line below always returns 32, as it should from counting the colours
    #on the scale manually
    unique_colours = len(colours1)

    #Range of values represented by each colour on the colour scale
    vals = np.linspace(-1, 1.2, unique_colours)

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
        #Edit the final matrix for image one, put the value in at all pixel
        #locations of that colour
        final_matrix1[mask1 != 0] = vals[i]
        mask2 = cv.inRange(im2, colours2[i] * 0.98, colours2[i] * 1.02)
        final_matrix2[mask2 != 0] = vals[i]
        #showim(hstack_images(mask1, mask2))

    value_delta_matrix = calculate_delta(final_matrix1, final_matrix2)

    max = np.max(value_delta_matrix)
    min = np.min(value_delta_matrix)
    all_delta_vals = np.unique(value_delta_matrix)
    all_delta_vals = np.sort(all_delta_vals)
    delta_vals = np.linspace(min_delta_val, max_delta_val, 32)

    value_delta_matrix = get_closest(delta_vals, value_delta_matrix)

    #Delta matrix will be constructed below to contain colours
    delta_matrix = np.ndarray((1080, 960, 3))

    for i, v in enumerate(delta_vals):
        #This should output an image composed of only the colours in the array colours1
        #There should be NO white pixels
        #I am very stupid, values in image matrix must be normalised between 0 and 1
        #MAX MAX MAX SUPER MAX MAX SUPER SUPER MAX
        delta_matrix[value_delta_matrix == v] = colours1[i] / 255

    scale_image = np.ndarray((1080, 960, 3))
    scale_image[:,:] = [0, 0, 0]
    scale_image[125:765, 80:150] = colour_scale1 / 255

    cv.putText(scale_image, str(max_delta_val), (80, 120), cv.FONT_HERSHEY_COMPLEX, 2, [1,1,1])
    cv.putText(scale_image, str(min_delta_val), (80, 800), cv.FONT_HERSHEY_COMPLEX, 2, [1,1,1])

    delta_image = cv.hconcat([scale_image, delta_matrix])
    return delta_image

def get_closest(values, matrix):

    #Get insert positions
    shape = np.shape(matrix)
    matrix = matrix.flatten()
    idxs = np.searchsorted(values, matrix, side="left")
    
    #Find indexes where previous index is closer
    #Hashtag Magic
    prev_idx_is_less = ((idxs == len(values))|(np.fabs(matrix - values[np.maximum(idxs-1, 0)]) < np.fabs(matrix - values[np.minimum(idxs, len(values)-1)])))
    idxs[prev_idx_is_less] -= 1
    flattened = values[idxs]
    rounded_matrix = np.reshape(flattened, shape)
    return rounded_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_one", type=str, help="Folder of baseline images")
    parser.add_argument("path_two", type=str, help="Folder of images to compare")
    args = parser.parse_args()
    path_one = os.path.abspath(args.path_one)
    path_two = os.path.abspath(args.path_two)

    delta_path = os.path.abspath(path_one + os.sep + ".." + os.sep + "delta")
    if not os.path.exists(delta_path):
        os.mkdir(delta_path)

    baseline_images = os.listdir(path_one)
    compare_images = os.listdir(path_two)

    for image in baseline_images:
        baseline = cv.imread(os.path.join(path_one, image))
        compare = cv.imread(os.path.join(path_two, image))

        im_number = int(image[5:-4]) #frame00084.png -> int("00084") -> 84

        if compare is None:
            print("%s could not be read in from the compare images, skipping." % image)
            continue

        delta_image = np.array(make_delta(baseline, compare))
        #showim(resize(delta_image))
        save_path = os.path.abspath(delta_path + os.sep + "delta%05d.png" % im_number)
        cv.imwrite(save_path, delta_image * 255)


    im1 = cv.imread("im1.png")
    im2 = cv.imread("im2.png")
