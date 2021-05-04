import math

import cv2
import numpy as np
from imutils import contours
import imutils

alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
            11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# input: ROI of First name on scan sheet
# output: String of the first name
def get_first_name(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cv2.imshow('Window', edged)
    cv2.waitKey(0)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break

    # apply Otsu's thresholding method to binarize the warped
    # piece of paper
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cv2.imshow('Thresholded Original', thresh)
    cv2.waitKey(0)

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    letters = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 10 and h >= 10 and ar >= 0.85 and ar <= 1.15:
            letters.append(c)
    original = image.copy()
    sortedContours = contours.sort_contours(letters, method="top-to-bottom")[0]
    fullySorted = []
    for i in range(0, len(fullySorted), 11):
        print(i)
        fullySorted.append(contours.sort_contours(sortedContours[i:i+11], method="left-to-right")[0])

    for i in range(0, len(letters)):
        cv2.drawContours(image, fullySorted[i], -1, (255, 0, 0), 3)
        cv2.imshow('letters', image)
        cv2.waitKey(0)


    for i in range(0, 26):
        cv2.drawContours(image, sortedContours[i], -1, (0, 255, 0), 3)
        cv2.imshow('Bubbles', image)
        cv2.waitKey(0)

    # cv2.drawContours(image, letters, -1, (0, 0, 255), 3)
    # cv2.imshow('Bubbles', image)
    # cv2.waitKey(0)


    print(letters)
    return 0


# input: ROI of the Last Name section on scan sheet
# output: String of the last name
def get_last_name(image):
    return 1


def get_uid(image):
    return 2


def get_additional_info(image):
    return 3
