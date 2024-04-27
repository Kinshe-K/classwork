import skimage

import numpy as np
from PIL import Image
import glob

test = []
train = []
imageDetails_list = []
finalTestCheck = []




def mse(imageA, imageB):
    #found from pyimagesearch.com!
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def compare_images(imageA, imageB):
    # Compare images just calles mse and checks to see if the images are identical or not
    #print("here")
    m = mse(imageA, imageB)

    if m == 0:
        return 1
    else:
        #print(m)
        return 0


def main():

    for finger in glob.glob(
            'test/*'):
        if finger.endswith('.txt'):
            imageDetails_list.append(finger[-12:])
        else:
            im = Image.open(finger)
            finalTestCheck.append(finger[-12:])
            test.append(im)

    for finger2 in glob.glob(
            'train/*'):
        if finger2.endswith('.txt'):
            None
            #imageDetails_list.append(finger2[-12:])
        else:
            im2 = Image.open(finger2)
            train.append(im2)

    trainlength = len(train)
    print("number of train - ", trainlength)

    testlength = len(test)
    print("Number of test - ", testlength)

    #print(test[0])

    #Image._show(train[0])
    #Image._show(train[1])
    #Image._show(train[2])

    # img = Image._show(image_list[0])
    # img.show()

    # print(imageDetails_list[0])
    # print(imageDetails_list[1])

    for everyFinger in train:
        test.append(everyFinger)
        #print()

    print(finalTestCheck[0])
    print(finalTestCheck[1])
    newLength = len(test)
    print("new test Length - ", newLength)
    x = 0
    while x < trainlength:
        i = 0


        while i < newLength:
            image1 = train[x]
            image1 = skimage.img_as_float(image1)
            #test.append(train[x])


            image2 = test[i]
            image2 = skimage.img_as_float(image2)

            valid = compare_images(image1,image2)
            #print("counter - ", i)

            if valid == 1:
                #Image._show(test[i])
                #print("yes")
                test.pop(i)
                i=0
                x+=1
                if x == trainlength:
                    break
                print("x= ", x)
                print("i= ", i )
            else:
                #do next iteration
                i += 1
                #print("next")

    print("final length of test - ", len(test))

    #for imageName in finalTestCheck:
    print("Image Name - ", finalTestCheck)

if __name__ == '__main__':
    main()