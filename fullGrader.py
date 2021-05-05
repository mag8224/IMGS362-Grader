from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
#import output
import output
import readAnswerSheet

# expected Input:
#   template = greyscale of the blank scan sheet to align the image with
#   image = greyscale of the student's scan sheet to get warped to match template image
def register_image(template, image, image_color):
    aligned = 0
    height, width = template.shape
    orb_detector = cv2.ORB_create(5000)

    # Find keypoints and descriptors.
    kp1, d1 = orb_detector.detectAndCompute(image, None)
    kp2, d2 = orb_detector.detectAndCompute(template, None)

    # Match features between the two images.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # matched = cv2.drawMatches(image, kp1, template, kp2, matches, None)
    # matched = imutils.resize(matched, width=1000)
    # cv2.imshow("Matches", matched)
    # cv2.waitKey(0)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Get homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # Warp input image to template
    aligned = cv2.warpPerspective(image_color, homography, (width, height))
    return aligned


def scale(template, image):
    scale_percent = 50  # percent of original size
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    aligned_show = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    template_show = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
    return aligned_show


def score(cnts, roi, thresh, answers, startQuestion):
    cnts1 = []
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(cnts), 5)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts1 = contours.sort_contours(cnts[i:i + 5])[0]
        bubbled = None

        # loop over the sorted contours
        for (j, c) in enumerate(cnts1):
            # construct a mask that reveals only the current
            # "bubble" for the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # bubble area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            # cv2.imshow('Mask', mask)
            # cv2.waitKey(0)
            total = cv2.countNonZero(mask)
            # if the current total has a larger number of total
            # non-zero pixels, then we are examining the currently
            # bubbled-in answer
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        # initialize the contour color and the index of the
        # *correct* answer
        color = (0, 0, 255)
        k = answers[q+startQuestion]
        # check to see if the bubbled answer is correct
        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1
        # draw the outline of the correct answer on the test
        cv2.drawContours(roi, [cnts1[k]], -1, color, 3)
        #cv2.imshow('Graded', roi)
        #cv2.waitKey(0)
    return correct


def gradeQuestions(roi, answers):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #get the thresholded roi
    thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #find all contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #get contours of a the specific aspect ratio
    bubbleContours = []
    questionCntsMod = []
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
            bubbleContours.append(c)

    for x in range(0, len(bubbleContours), 30):
        questionCntsSortedLR = bubbleContours[x:x + 30]
        # cv2.drawContours(image, questionCntsSortedLR, -1, (0, 100, 255), 3)
        # cv2.imshow('Bubbles', image)
        # cv2.waitKey(0)
        questionCntsSortedLR = contours.sort_contours(questionCntsSortedLR, method="left-to-right")[0]
        questionCntsMod += questionCntsSortedLR

    questionCnts1 = []
    questionCnts2 = []
    questionCnts3 = []
    questionCnts4 = []
    questionCnts5 = []
    questionCnts6 = []

    for x in range(0, len(bubbleContours), 30):
        questionCnts1 += questionCntsMod[x:x + 5]
        questionCnts2 += questionCntsMod[x + 5:x + 10]
        questionCnts3 += questionCntsMod[x+10:x+15]
        questionCnts4 += questionCntsMod[x + 15:x + 20]
        questionCnts5 += questionCntsMod[x + 20:x + 25]
        questionCnts6 += questionCntsMod[x + 25:x + 30]

    questionCnts1 = contours.sort_contours(questionCnts1, method="top-to-bottom")[0]
    questionCnts2 = contours.sort_contours(questionCnts2, method="top-to-bottom")[0]
    questionCnts3 = contours.sort_contours(questionCnts3, method="top-to-bottom")[0]
    questionCnts4 = contours.sort_contours(questionCnts4, method="top-to-bottom")[0]
    questionCnts5 = contours.sort_contours(questionCnts5, method="top-to-bottom")[0]
    questionCnts6 = contours.sort_contours(questionCnts6, method="top-to-bottom")[0]
    correct = 0
    scoreCol1 = score(questionCnts1, roi, thresh, answers, 1)
    correct += scoreCol1
    if(len(answers)> 25):
        scoreCol2 = score(questionCnts2, roi, thresh, answers, 26)
        correct += scoreCol2
    if(len(answers)> 50):
        scoreCol3 = score(questionCnts3, roi, thresh, answers, 51)
        correct += scoreCol3
    if(len(answers)> 75):
        scoreCol4 = score(questionCnts4, roi, thresh, answers, 76)
        correct += scoreCol4
    if(len(answers) > 100):
        scoreCol5 = score(questionCnts5, roi, thresh, answers, 101)
        correct += scoreCol5
    if(len(answers)> 125):
        scoreCol6 = score(questionCnts6, roi, thresh, answers, 126)
        correct += scoreCol6
    return correct / len(answers) * 100


def readLetter(roi, thresh, cnts):
    LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U",
               "V", "W", "X", "Y", 'Z']
    T = 120
    bubbled = None
    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        #cv2.imshow("mask", mask)
        #cv2.waitKey(0)
        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        total = cv2.countNonZero(mask)
        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if total > T:
            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
                return LETTERS[j]
    return ""

def findLastName(roi):

    gray_ln = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh_ln = cv2.threshold(gray_ln, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts_ln = cv2.findContours(thresh_ln.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts_ln = imutils.grab_contours(cnts_ln)
    lastNameCnts = []
    # loop over the contours
    for c in cnts_ln:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 8 and h >= 8 and ar >= 0.8 and ar <= 1.2:
            lastNameCnts.append(c)
    LNcnts = []
    for x in range(0, len(lastNameCnts), 11):
        SortedLR = lastNameCnts[x:x + 11]
        SortedLR = contours.sort_contours(SortedLR, method="left-to-right")[0]
        LNcnts += SortedLR

    LNCnts1 = []
    LNCnts2 = []
    LNCnts3 = []
    LNCnts4 = []
    LNCnts5 = []
    LNCnts6 = []
    LNCnts7 = []
    LNCnts8 = []
    LNCnts9 = []
    LNCnts10 = []
    LNCnts11 = []

    for x in range(0, len(lastNameCnts), 11):
        LNCnts1 += LNcnts[x:x + 1]
        LNCnts2 += LNcnts[x + 1:x + 2]
        LNCnts3 += LNcnts[x + 2:x + 3]
        LNCnts4 += LNcnts[x + 3:x + 4]
        LNCnts5 += LNcnts[x + 4:x + 5]
        LNCnts6 += LNcnts[x + 5:x + 6]
        LNCnts7 += LNcnts[x + 6:x + 7]
        LNCnts8 += LNcnts[x + 7:x + 8]
        LNCnts9 += LNcnts[x + 8:x + 9]
        LNCnts10 += LNcnts[x + 9:x + 10]
        LNCnts11 += LNcnts[x + 10:x + 11]

    LNCnts1 = contours.sort_contours(LNCnts1, method="top-to-bottom")[0]
    LNCnts2 = contours.sort_contours(LNCnts2, method="top-to-bottom")[0]
    LNCnts3 = contours.sort_contours(LNCnts3, method="top-to-bottom")[0]
    LNCnts4 = contours.sort_contours(LNCnts4, method="top-to-bottom")[0]
    LNCnts5 = contours.sort_contours(LNCnts5, method="top-to-bottom")[0]
    LNCnts6 = contours.sort_contours(LNCnts6, method="top-to-bottom")[0]
    LNCnts7 = contours.sort_contours(LNCnts7, method="top-to-bottom")[0]
    LNCnts8 = contours.sort_contours(LNCnts8, method="top-to-bottom")[0]
    LNCnts9 = contours.sort_contours(LNCnts9, method="top-to-bottom")[0]
    LNCnts10 = contours.sort_contours(LNCnts10, method="top-to-bottom")[0]
    LNCnts11 = contours.sort_contours(LNCnts11, method="top-to-bottom")[0]

    LNContoursList = [LNCnts1, LNCnts2, LNCnts3, LNCnts4, LNCnts5, LNCnts6, LNCnts7, LNCnts8, LNCnts9, LNCnts10,
                      LNCnts11]
    LNContoursMat = np.array(LNContoursList, dtype=object)

    letter1 = readLetter(roi, thresh_ln, LNCnts1)
    letter2 = readLetter(roi, thresh_ln, LNCnts2)
    letter3 = readLetter(roi, thresh_ln, LNCnts3)
    letter4 = readLetter(roi, thresh_ln, LNCnts4)
    letter5 = readLetter(roi, thresh_ln, LNCnts5)
    letter6 = readLetter(roi, thresh_ln, LNCnts6)
    letter7 = readLetter(roi, thresh_ln, LNCnts7)
    letter8 = readLetter(roi, thresh_ln, LNCnts8)
    letter9 = readLetter(roi, thresh_ln, LNCnts9)
    letter10 = readLetter(roi, thresh_ln, LNCnts10)
    letter11 = readLetter(roi, thresh_ln, LNCnts11)

    word = letter1 + letter2 + letter3 + letter4 + letter5 + letter6 + letter7 + letter8 + letter9 + letter10 + letter11
    return word

def findFirstName(roi):

    gray_ln = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh_ln = cv2.threshold(gray_ln, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts_fn = cv2.findContours(thresh_ln.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts_ln = imutils.grab_contours(cnts_fn)
    firstNameCnts = []
    # loop over the contours
    for c in cnts_ln:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 8 and h >= 8 and ar >= 0.8 and ar <= 1.2:
            firstNameCnts.append(c)
    FNcnts = []
    for x in range(0, len(firstNameCnts), 11):
        SortedLR = firstNameCnts[x:x + 11]
        SortedLR = contours.sort_contours(SortedLR, method="left-to-right")[0]
        FNcnts += SortedLR

    FNCnts1 = []
    FNCnts2 = []
    FNCnts3 = []
    FNCnts4 = []
    FNCnts5 = []
    FNCnts6 = []
    FNCnts7 = []
    FNCnts8 = []
    FNCnts9 = []


    for x in range(0, len(firstNameCnts), 11):
        FNCnts1 += FNcnts[x:x + 1]
        FNCnts2 += FNcnts[x + 1:x + 2]
        FNCnts3 += FNcnts[x + 2:x + 3]
        FNCnts4 += FNcnts[x + 3:x + 4]
        FNCnts5 += FNcnts[x + 4:x + 5]
        FNCnts6 += FNcnts[x + 5:x + 6]
        FNCnts7 += FNcnts[x + 6:x + 7]
        FNCnts8 += FNcnts[x + 7:x + 8]
        FNCnts9 += FNcnts[x + 8:x + 9]


    FNCnts1 = contours.sort_contours(FNCnts1, method="top-to-bottom")[0]
    FNCnts2 = contours.sort_contours(FNCnts2, method="top-to-bottom")[0]
    FNCnts3 = contours.sort_contours(FNCnts3, method="top-to-bottom")[0]
    FNCnts4 = contours.sort_contours(FNCnts4, method="top-to-bottom")[0]
    FNCnts5 = contours.sort_contours(FNCnts5, method="top-to-bottom")[0]
    FNCnts6 = contours.sort_contours(FNCnts6, method="top-to-bottom")[0]
    FNCnts7 = contours.sort_contours(FNCnts7, method="top-to-bottom")[0]
    FNCnts8 = contours.sort_contours(FNCnts8, method="top-to-bottom")[0]
    FNCnts9 = contours.sort_contours(FNCnts9, method="top-to-bottom")[0]

    letter1 = readLetter(roi, thresh_ln, FNCnts1)
    letter2 = readLetter(roi, thresh_ln, FNCnts2)
    letter3 = readLetter(roi, thresh_ln, FNCnts3)
    letter4 = readLetter(roi, thresh_ln, FNCnts4)
    letter5 = readLetter(roi, thresh_ln, FNCnts5)
    letter6 = readLetter(roi, thresh_ln, FNCnts6)
    letter7 = readLetter(roi, thresh_ln, FNCnts7)
    letter8 = readLetter(roi, thresh_ln, FNCnts8)
    letter9 = readLetter(roi, thresh_ln, FNCnts9)

    word = letter1 + letter2 + letter3 + letter4 + letter5 + letter6 + letter7 + letter8 + letter9
    return word

def readNumber(roi, thresh, cnts ):
    T = 120
    NUMBERS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    bubbled = None
    letter = ""
    # loop over the sorted contours
    for (j, c) in enumerate(cnts):
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        # cv2.imshow('Mask', mask)
        # cv2.waitKey(0)
        total = cv2.countNonZero(mask)
        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0] and total > T:
            bubbled = (total, j)
            letter = NUMBERS[j]
    return letter



def findOtherInfo(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts_uid = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts_uid = imutils.grab_contours(cnts_uid)
    uidCnts = []

    # loop over the contours
    for c in cnts_uid:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 8 and h >= 8 and ar >= 0.8 and ar <= 1.2:
            uidCnts.append(c)

    uidCntsSorted = []
    for x in range(0, len(uidCnts), 9):
        sortedLR = uidCnts[x:x+9]
        sortedLR = contours.sort_contours(sortedLR, method="left-to-right")[0]
        uidCntsSorted += sortedLR
    uidCnts1 = []
    uidCnts2 = []
    uidCnts3 = []
    uidCnts4 = []
    uidCnts5 = []
    uidCnts6 = []
    uidCnts7 = []
    uidCnts8 = []
    uidCnts9 = []

    for x in range(0, len(uidCnts), 9):
        uidCnts1 += uidCntsSorted[x:x + 1]
        uidCnts2 += uidCntsSorted[x + 1:x + 2]
        uidCnts3 += uidCntsSorted[x + 2:x + 3]
        uidCnts4 += uidCntsSorted[x + 3:x + 4]
        uidCnts5 += uidCntsSorted[x + 4:x + 5]
        uidCnts6 += uidCntsSorted[x + 5:x + 6]
        uidCnts7 += uidCntsSorted[x + 6:x + 7]
        uidCnts8 += uidCntsSorted[x + 7:x + 8]
        uidCnts9 += uidCntsSorted[x + 8:x + 9]


    uidCnts1 = contours.sort_contours(uidCnts1, method="top-to-bottom")[0]
    uidCnts2 = contours.sort_contours(uidCnts2, method="top-to-bottom")[0]
    uidCnts3 = contours.sort_contours(uidCnts3, method="top-to-bottom")[0]
    uidCnts4 = contours.sort_contours(uidCnts4, method="top-to-bottom")[0]
    uidCnts5 = contours.sort_contours(uidCnts5, method="top-to-bottom")[0]
    uidCnts6 = contours.sort_contours(uidCnts6, method="top-to-bottom")[0]
    uidCnts7 = contours.sort_contours(uidCnts7, method="top-to-bottom")[0]
    uidCnts8 = contours.sort_contours(uidCnts8, method="top-to-bottom")[0]
    uidCnts9 = contours.sort_contours(uidCnts9, method="top-to-bottom")[0]

    letter1 = readNumber(roi, thresh, uidCnts1)
    letter2 = readLetter(roi, thresh, uidCnts2)
    letter3 = readLetter(roi, thresh, uidCnts3)
    letter4 = readLetter(roi, thresh, uidCnts4)
    letter5 = readLetter(roi, thresh, uidCnts5)
    letter6 = readLetter(roi, thresh, uidCnts6)
    letter7 = readLetter(roi, thresh, uidCnts7)
    letter8 = readLetter(roi, thresh, uidCnts8)
    letter9 = readLetter(roi, thresh, uidCnts9)

    uid = letter1 + letter2 + letter3 + letter4 + letter5 + letter6 + letter7 + letter8 + letter9
    return uid

def read_scan_sheet(template, image, answers):
    template_color = template
    image_color = image

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    height, width = template_gray.shape
    print('Template Shape', height, width)
    aligned_color = register_image(template_gray, image_gray, image_color)
    aligned_color = scale(template, aligned_color)
    cv2.imshow("aligned Image", aligned_color)
    cv2.waitKey(0)

    ## Take the roi areas for each of the bubbled sections ##
    questionsROI = aligned_color[540:1000, 95:750]
    lastNameROI = aligned_color[70:500, 95:300]
    firstNameROI = aligned_color[70:500, 320:500]
    uidROI = aligned_color[65:240, 510:700]
    additionalInfoROI = aligned_color[300:500, 500:700]

    # begin actually reading the bubbles
    score = gradeQuestions(questionsROI, answers)
    print(score)
    lastName = findLastName(lastNameROI)
    print(lastName)
    firstName = findFirstName(firstNameROI)
    print(firstName)
    uid = findOtherInfo(uidROI)
    print(uid)
    additional = findOtherInfo(additionalInfoROI)
    print(additional)
    return score, lastName, firstName, uid, additional


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    #ap.add_argument("-k", "--key", required=True, help="Test Answer Key")
    ap.add_argument("-i", "--image", required=True, help="Image of Student Test")
    ap.add_argument("-k", "--key", required=True, help="image of test key")
    args = vars(ap.parse_args())
    print(args)
    # Read in sheet to be graded
    original = cv2.imread(args["image"])
    image = cv2.imread(args["image"])
    key = cv2.imread(args["key"])
    template = cv2.imread("original_scan_sheet.png")
    #Read in answers
    answers = readAnswerSheet.readAnswers(key,template) #answers have been read in as a dictionary
    print(answers)
    #read scan sheet
    score, lastName, firstName, uid, additional = read_scan_sheet(template, image, answers)
    print(score, lastName, firstName, uid, additional)
    output.create_output("results.csv")
    filename = "results.csv"
    output.append_score(str(score), filename)
    output.append_score("," + str(lastName), filename)
    output.append_score("," + str(firstName), filename)
    output.append_score("," + str(uid), filename)
    output.append_score("," + str(additional), filename)




