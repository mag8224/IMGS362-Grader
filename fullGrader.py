from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
#import output
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

def gradeQuestions(roi, answers):

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
    score = gradeQuestions(questionsROI, answers)

    # Scale image and template


    return 0

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
    read_scan_sheet(template, image, answers)



