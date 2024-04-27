from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
#import keras
import skimage
import glob
from PIL import Image

test = []
train = []
finalTestCheck = []



def compareImages(image1, image2):
    #print("here")
    #Compares the structural similarity Index(SSIM) between two images
    score = compare_ssim(image1,image2)
    #print("score - ", score)
    if score == 1:
        return 1
    else:
        return 0

def main():
    print("Start")

    for finger in glob.glob(
            'test/*'):
        im = Image.open(finger)
        finalTestCheck.append(finger[-12:])
        test.append(im)

    for finger2 in glob.glob(
            'train/*'):
        #if filename.endswith('.txt'):
         #   imageDetails_list.append(filename[-12:])
        #else:
            im2 = Image.open(finger2)
            train.append(im2)

    trainlength = len(train)
    print("number of train - ", trainlength)

    testlength = len(test)
    print("Number of test - ", testlength)

    for everyFinger in train:
        test.append(everyFinger)
        # print()

    print(finalTestCheck[0])
    print(finalTestCheck[1])


    #image1 = skimage.img_as_float(test[0])
    #image2 = skimage.img_as_float(test[0])
    #compareImages(image1, image2)


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

            valid = compareImages(image1,image2)
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




