from imutils import contours
import numpy as np
#import argparse
import imutils
import cv2

def readAnswers(image, template):
	# Convert to gray
	image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	height, width = template_gray.shape

	# Create ORB detector with 5000 features.
	orb_detector = cv2.ORB_create(5000)

	# Find keypoints and descriptors.
	kp1, d1 = orb_detector.detectAndCompute(image_gray, None)
	kp2, d2 = orb_detector.detectAndCompute(template_gray, None)

	# Match features between the two images.
	matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

	# Match the two sets of descriptors.
	matches = matcher.match(d1, d2)

	# Sort matches on the basis of their Hamming distance.
	matches.sort(key=lambda x: x.distance)

	# Take the top 90 % matches forward.
	matches = matches[:int(len(matches) * 90)]
	no_of_matches = len(matches)

	matched = cv2.drawMatches(image, kp1, template, kp2, matches, None)
	matched = imutils.resize(matched, width=1000)
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
	aligned = cv2.warpPerspective(image, homography, (width, height))


	# Scale image and template
	scale_percent = 50  # percent of original size
	width = int(image.shape[1] * scale_percent / 100)
	height = int(image.shape[0] * scale_percent / 100)
	dim = (width, height)

	aligned = cv2.resize(aligned, dim, interpolation=cv2.INTER_AREA)
	template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
	#
	# cv2.imshow("Image", aligned)
	# cv2.imshow("Template", template)
	# cv2.waitKey(0)
	# cv2.waitKey(0)


	# resize both the aligned and template images so we can easily
	# visualize them on our screen
	aligned_show = imutils.resize(aligned, width=600)
	template_show = imutils.resize(template, width=600)
	# our first output visualization of the image alignment will be a
	# side-by-side comparison of the output aligned image and the
	# template
	stacked = np.hstack([aligned_show, template_show])
	#
	# cv2.imshow("Image Alignment Stacked", stacked)
	# cv2.waitKey(0)

	# print(aligned.shape)
	# Crop the answer section out of registered image
	image = aligned[800:1450, 100:1100]
	# cv2.imshow("aligned", aligned)
	# cv2.imshow("cropped", image)
	# cv2.waitKey(0)


	# load the image, convert it to grayscale, blur it
	# slightly, then find edges

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(blurred, 75, 200)
	# cv2.imshow('Window', edged)
	# cv2.waitKey(0)


	# find contours in the edge map, then initialize
	# the contour that corresponds to the document
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	docCnt = None
	# ensure that at least one contour was found
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

	#
	# cv2.imshow('Thresholded Original', thresh)
	# cv2.waitKey(0)


	# find contours in the thresholded image, then initialize
	# the list of contours that correspond to questions
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	questionCnts = []

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
			questionCnts.append(c)

	print(len(questionCnts))
	original_image = image.copy()

	cv2.drawContours(image, questionCnts, -1, (0, 0, 255), 3)
	# cv2.imshow('Bubbles', image)
	# cv2.waitKey(0)
	questionCntsMod = []
	for x in range(0, len(questionCnts), 30): #30 comes from 5 bubbles per row of a question, with 6 columns across
		questionCntsSortedLR = questionCnts[x:x+30]
		questionCntsSortedLR = contours.sort_contours(questionCntsSortedLR, method="left-to-right")[0]
		questionCntsMod += questionCntsSortedLR

	questionCnts1 = []
	questionCnts2 = []
	questionCnts3 = []
	questionCnts4 = []
	questionCnts5 = []
	questionCnts6 = []


	for x in range(0, len(questionCnts), 30):

		questionCnts1 += questionCntsMod[x:x + 5]
		questionCnts2 += questionCntsMod[x+5:x+10]
		questionCnts3 += questionCntsMod[x+10:x+15]
		questionCnts4 += questionCntsMod[x+15:x+20]
		questionCnts5 += questionCntsMod[x+20:x+25]
		questionCnts6 += questionCntsMod[x+25:x+30]

	# cv2.drawContours(image, questionCnts2, -1, (0, 0, 255), 3)
	# cv2.imshow('Bubbles', image)
	# cv2.waitKey(0)

	# sort the question contours top-to-bottom, then initialize
	# the total number of correct answers
	questionCnts1 = contours.sort_contours(questionCnts1, method="top-to-bottom")[0]
	questionCnts2 = contours.sort_contours(questionCnts2, method="top-to-bottom")[0]
	questionCnts3 = contours.sort_contours(questionCnts3, method="top-to-bottom")[0]
	questionCnts4 = contours.sort_contours(questionCnts4, method="top-to-bottom")[0]
	questionCnts5 = contours.sort_contours(questionCnts5, method="top-to-bottom")[0]
	questionCnts6 = contours.sort_contours(questionCnts6, method="top-to-bottom")[0]

	correct = 0
	# each question has 5 possible answers, to loop over the
	# question in batches of 5

	cnts1 = []
	answers = {}
	ANSWER_KEY_COL1 = {}
	import output
	file = "answerKey.csv"
	output.create_output("answerKey.csv")
	output.append_score("Answer Key", file)
	output.newLine(file)


	for (q, i) in enumerate(np.arange(0, len(questionCnts1), 5)):
		#q is the question number
		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts1 = contours.sort_contours(questionCnts1[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts1):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			#cv2.imshow('mask', mask)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+1] = j
				text = str(q + 1) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL1[q] = j

	cnts2 = []
	ANSWER_KEY_COL2 = {}
	for (q, i) in enumerate(np.arange(0, len(questionCnts2), 5)):
		#q is the question number

		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts2 = contours.sort_contours(questionCnts2[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts2):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			cv2.imshow('mask', mask)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+26] = j
				text = str(q + 26) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL2[q] = j



	cnts3 = []
	ANSWER_KEY_COL3 = {}

	for (q, i) in enumerate(np.arange(0, len(questionCnts3), 5)):
		#q is the question number

		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts3 = contours.sort_contours(questionCnts3[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts3):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			cv2.imshow('mask', mask)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+51] = j
				text = str(q + 51) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL3[q] = j


	cnts4 = []
	ANSWER_KEY_COL4 = {}
	for (q, i) in enumerate(np.arange(0, len(questionCnts4), 5)):
		#q is the question number

		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts4 = contours.sort_contours(questionCnts4[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts4):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			cv2.imshow('mask', mask)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+76] = j
				text = str(q + 76) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL4[q] = j



	cnts5 = []
	ANSWER_KEY_COL5 = {}

	for (q, i) in enumerate(np.arange(0, len(questionCnts5), 5)):
		#q is the question number

		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts5 = contours.sort_contours(questionCnts5[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts5):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			cv2.imshow('mask', mask)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+101] = j
				text = str(q + 101) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL5[q] = j


	cnts6 = []
	ANSWER_KEY_COL6 = {}

	for (q, i) in enumerate(np.arange(0, len(questionCnts6), 5)):
		#q is the question number

		# sort the contours for the current question from
		# left to right, then initialize the index of the
		# bubbled answer
		cnts6 = contours.sort_contours(questionCnts6[i:i + 5])[0]
		bubbled = None

		# loop over the sorted contours
		for (j, c) in enumerate(cnts6):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			cv2.imshow('mask', mask)

			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh, thresh, mask=mask) #idk why this is thresh twice
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > 300 or (bubbled is not None and total > bubbled[0]):
				bubbled = (total, j)
				answers[q+126] = j
				text = str(q + 126) + "," + str(j) + "\n"
				output.append_score(text, file)
				ANSWER_KEY_COL6[q] = j
	return answers


