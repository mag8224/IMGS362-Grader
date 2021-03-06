from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import csv

# Read in filename of document to be graded from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
args = vars(ap.parse_args())

files = ["answer_sheets_Page_2.png","answer_sheets_Page_3.png", "answer_sheets_Page_4.png",
	 "answer_sheets_Page_5.png","answer_sheets_Page_6.png","answer_sheets_Page_7.png",
	 "answer_sheets_Page_8.png","answer_sheets_Page_9.png"]

with open('grades.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	writer.writerow(["LAST NAME", "FIRST NAME", "SCORE", "UID", "ADDITIONAL INFO"])
	output_mat = []
	for f in files:

		# Read in document to be graded as image
		original = cv2.imread(f)
		image = cv2.imread(f)

		# Read in reference template of scan sheet
		template = cv2.imread("original_scan_sheet.png")

		# Convert to gray
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		height, width = template_gray.shape
		print(width)
		print(height)

		##################### IMAGE REGISTRATION ##########################

		# Adapted from https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/

		# Use ORB detection
		orb_detector = cv2.ORB_create(5000)

		# Compute keypoints and descriptors
		kp1, d1 = orb_detector.detectAndCompute(image_gray, None)
		kp2, d2 = orb_detector.detectAndCompute(template_gray, None)

		# Match features
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

		# Match descriptors
		matches = matcher.match(d1, d2)

		# Sort matches by distance
		matches.sort(key=lambda x: x.distance)

		# Keep best 90 % matches
		matches = matches[:int(len(matches) * 90)]
		no_of_matches = len(matches)

		# Draw matched between input image and template
		matched = cv2.drawMatches(image, kp1, template, kp2, matches, None)
		matched = imutils.resize(matched, width=1000)
		# cv2.imshow("Matches", matched)
		# cv2.waitKey(0)

		# Place match locations into new arrays
		p1 = np.zeros((no_of_matches, 2))
		p2 = np.zeros((no_of_matches, 2))
		for i in range(len(matches)):
			p1[i, :] = kp1[matches[i].queryIdx].pt
			p2[i, :] = kp2[matches[i].trainIdx].pt

		# Get homography matrix.
		homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

		# Warp input image to template
		aligned = cv2.warpPerspective(image, homography, (width, height))



		##################### GRADING SCHEME ##########################

		# Adapted from https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/

		# Scale image and template
		scale_percent = 50  # percent of original size
		width = int(template.shape[1] * scale_percent / 100)
		height = int(template.shape[0] * scale_percent / 100)
		dim = (width, height)

		aligned = cv2.resize(aligned, dim, interpolation=cv2.INTER_AREA)
		template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

		# Make copy of aligned image
		aligned_ln = aligned.copy()

		# Resize both aligned and template images for viewing
		aligned_show = imutils.resize(aligned, width=600)
		template_show = imutils.resize(template, width=600)
		# Place aligned image and template side by side
		stacked = np.hstack([aligned_show, template_show])

		# Display aligned image with template
		# cv2.imshow("Image Alignment Stacked", stacked)
		# cv2.waitKey(0)

		# Crop the answer section out of registered image
		image = aligned[785:1420, 100:1100]

		# Define answer keys for each column of questions
		ANSWER_KEY = {
			0: 2, 1: 1, 2: 2, 3: 1, 4: 0, 5: 1, 6: 2, 7: 2, 8: 1, 9: 3, 10: 3, 11: 2, 12: 3, 13: 1, 14: 3, 15: 0,
			16: 1, 17: 2, 18: 1, 19: 3, 20: 2, 21: 1, 22: 2, 23: 1, 24: 0}

		ANSWER_KEY_COL2 = {
			0: 2, 1: 1, 2: 3, 3: 0, 4: 1, 5: 2, 6: 0, 7: 1, 8: 2, 9: 1, 10: 2, 11: 0, 12: 1, 13: 3, 14: 2, 15: 1,
			16: 2, 17: 2, 18: 1, 19: 3, 20: 0, 21: 1, 22: 2, 23: 1, 24: 3}

		# Convert cropped image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# Binarize using Otsu's method
		thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		# Display thresholded image
		# cv2.imshow('Thresholded Original', thresh)
		# cv2.waitKey(0)


		# detect contours in thresholded image
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
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

		# Print number of detected answer bubble contours (should be 750)
		print(len(questionCnts))

		# Draw and display detected contours
		#cv2.drawContours(image, cnts, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', image)
		#cv2.waitKey(0)


		# Sort each set of 30 contours (the number of bubbles horizontally across) from left to right
		questionCntsMod = []
		for x in range(0, len(questionCnts), 30):
			questionCntsSortedLR = questionCnts[x:x+30]
			questionCntsSortedLR = contours.sort_contours(questionCntsSortedLR, method="left-to-right")[0]
			questionCntsMod += questionCntsSortedLR

		# Separate the contours into separate columns of answer bubbles for each column of questions
		questionCnts1 = []
		questionCnts2 = []
		for x in range(0, len(questionCnts), 30):
			questionCnts1 += questionCntsMod[x:x + 5]
			questionCnts2 += questionCntsMod[x+5:x+10]

		# cv2.drawContours(image, questionCnts2, -1, (0, 0, 255), 3)
		# cv2.imshow('Bubbles', image)
		# cv2.waitKey(0)

		# sort the question contours top-to-bottom
		questionCnts1 = contours.sort_contours(questionCnts1, method="top-to-bottom")[0]
		questionCnts2 = contours.sort_contours(questionCnts2, method="top-to-bottom")[0]

		# initialize number of correct answers
		correct = 0
		# list to hold sorted contours
		cnts1 = []
		# loops over contours in intervals of 5 (5 bubbles per question)
		for (q, i) in enumerate(np.arange(0, len(questionCnts1), 5)):
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
			k = ANSWER_KEY[q]
			# check to see if the bubbled answer is correct
			if k == bubbled[1]:
				color = (0, 255, 0)
				correct += 1
			# draw the outline of the correct answer on the test
			cv2.drawContours(image, [cnts1[k]], -1, color, 3)

		cnts2 = []
		for (q, i) in enumerate(np.arange(0, len(questionCnts2), 5)):
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
				# apply the mask to the thresholded image, then
				# count the number of non-zero pixels in the
				# bubble area
				mask = cv2.bitwise_and(thresh, thresh, mask=mask)
				total = cv2.countNonZero(mask)
				# if the current total has a larger number of total
				# non-zero pixels, then we are examining the currently
				# bubbled-in answer
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)


			# initialize the contour color and the index of the
			# *correct* answer
			color = (0, 0, 255)
			k = ANSWER_KEY_COL2[q]
			# check to see if the bubbled answer is correct
			if k == bubbled[1]:
				color = (0, 255, 0)
				correct += 1
			# draw the outline of the correct answer on the test
			cv2.drawContours(image, [cnts2[k]], -1, color, 3)


		# cv2.imshow('Correct', image)
		# cv2.waitKey(0)

		aligned_show = imutils.resize(aligned, width=600)
		# cv2.imshow('Aligned', aligned_show)
		# cv2.waitKey(0)

		# Calculate score and write it at top of page
		score = (correct / 50.0) * 100
		print("[INFO] score: {:.2f}%".format(score))
		cv2.putText(aligned_show, "{:.2f}%".format(score), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
		cv2.imshow("Original", aligned_show)
		cv2.waitKey(0)


		LETTERS = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y",'Z']
		T = 130

		last_name_image = aligned_ln[105:750, 140:430]
		#cv2.imshow("Last Name Region", last_name_image)
		#cv2.waitKey(0)

		gray_ln = cv2.cvtColor(last_name_image, cv2.COLOR_BGR2GRAY)
		thresh_ln = cv2.threshold(gray_ln, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		#cv2.imshow("Thresh", thresh_ln)
		#cv2.waitKey(0)

		# find contours in the thresholded image, then initialize
		# the list of contours that correspond to questions
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

		#cv2.drawContours(last_name_image, lastNameCnts, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNcnts = []
		for x in range(0, len(lastNameCnts), 11):
			SortedLR = lastNameCnts[x:x+11]
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

		for x in range(0, len(questionCnts), 11):
			LNCnts1 += LNcnts[x:x + 1]
			LNCnts2 += LNcnts[x+1:x+2]
			LNCnts3 += LNcnts[x+2:x+3]
			LNCnts4 += LNcnts[x+3:x+4]
			LNCnts5 += LNcnts[x+4:x+5]
			LNCnts6 += LNcnts[x+5:x+6]
			LNCnts7 += LNcnts[x+6:x+7]
			LNCnts8 += LNcnts[x+7:x+8]
			LNCnts9 += LNcnts[x+8:x+9]
			LNCnts10 += LNcnts[x+9:x+10]
			LNCnts11 += LNcnts[x+10:x+11]



		#cv2.drawContours(last_name_image, LNCnts3, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

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

		LNContoursList = [LNCnts1, LNCnts2, LNCnts3, LNCnts4,LNCnts5,LNCnts6,LNCnts7,LNCnts8,LNCnts9,LNCnts10,LNCnts11]
		LNContoursMat = np.array(LNContoursList)
		# print(LNContoursMat)

		letter = ""
		lastname = ""
		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts1):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts2):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter


		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts3):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts4):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts5):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts6):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts7):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts8):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts9):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter





		print("")
		print("Last Name:")
		print(lastname)

		last_name = lastname


		###### FIRST NAME ###############
		last_name_image = aligned_ln[105:750, 475:710]
		# cv2.imshow("First Name Region", last_name_image)
		# cv2.waitKey(0)

		gray_ln = cv2.cvtColor(last_name_image, cv2.COLOR_BGR2GRAY)
		thresh_ln = cv2.threshold(gray_ln, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		#cv2.imshow("Thresh", thresh_ln)
		#cv2.waitKey(0)

		# find contours in the thresholded image, then initialize
		# the list of contours that correspond to questions
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

		#cv2.drawContours(last_name_image, lastNameCnts, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNcnts = []
		for x in range(0, len(lastNameCnts), 9):
			SortedLR = lastNameCnts[x:x+9]
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

		for x in range(0, len(questionCnts), 9):
			LNCnts1 += LNcnts[x:x + 1]
			LNCnts2 += LNcnts[x+1:x+2]
			LNCnts3 += LNcnts[x+2:x+3]
			LNCnts4 += LNcnts[x+3:x+4]
			LNCnts5 += LNcnts[x+4:x+5]
			LNCnts6 += LNcnts[x+5:x+6]
			LNCnts7 += LNcnts[x+6:x+7]
			LNCnts8 += LNcnts[x+7:x+8]
			LNCnts9 += LNcnts[x+8:x+9]



		#cv2.drawContours(last_name_image, LNCnts3, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNCnts1 = contours.sort_contours(LNCnts1, method="top-to-bottom")[0]
		LNCnts2 = contours.sort_contours(LNCnts2, method="top-to-bottom")[0]
		LNCnts3 = contours.sort_contours(LNCnts3, method="top-to-bottom")[0]
		LNCnts4 = contours.sort_contours(LNCnts4, method="top-to-bottom")[0]
		LNCnts5 = contours.sort_contours(LNCnts5, method="top-to-bottom")[0]
		LNCnts6 = contours.sort_contours(LNCnts6, method="top-to-bottom")[0]
		LNCnts7 = contours.sort_contours(LNCnts7, method="top-to-bottom")[0]
		LNCnts8 = contours.sort_contours(LNCnts8, method="top-to-bottom")[0]
		LNCnts9 = contours.sort_contours(LNCnts9, method="top-to-bottom")[0]

		LNContoursList = [LNCnts1, LNCnts2, LNCnts3, LNCnts4,LNCnts5,LNCnts6,LNCnts7,LNCnts8,LNCnts9]
		LNContoursMat = np.array(LNContoursList)
		# print(LNContoursMat)

		letter = ""
		lastname = ""
		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts1):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter


		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts2):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter


		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts3):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts4):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts5):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts6):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts7):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts8):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts9):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0] and total > 200:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		print("First Name")
		print(lastname)

		first_name = lastname


		###### UID ###############

		LETTERS = ["0", "1","2","3","4","5","6","7","8","9"]

		last_name_image = aligned_ln[105:380, 753:985]
		# cv2.imshow("UID Region", last_name_image)
		# cv2.waitKey(0)

		gray_ln = cv2.cvtColor(last_name_image, cv2.COLOR_BGR2GRAY)
		thresh_ln = cv2.threshold(gray_ln, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		#cv2.imshow("Thresh", thresh_ln)
		#cv2.waitKey(0)

		# find contours in the thresholded image, then initialize
		# the list of contours that correspond to questions
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

		#cv2.drawContours(last_name_image, lastNameCnts, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNcnts = []
		for x in range(0, len(lastNameCnts), 9):
			SortedLR = lastNameCnts[x:x+9]
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

		for x in range(0, len(questionCnts), 9):
			LNCnts1 += LNcnts[x:x + 1]
			LNCnts2 += LNcnts[x+1:x+2]
			LNCnts3 += LNcnts[x+2:x+3]
			LNCnts4 += LNcnts[x+3:x+4]
			LNCnts5 += LNcnts[x+4:x+5]
			LNCnts6 += LNcnts[x+5:x+6]
			LNCnts7 += LNcnts[x+6:x+7]
			LNCnts8 += LNcnts[x+7:x+8]
			LNCnts9 += LNcnts[x+8:x+9]



		#cv2.drawContours(last_name_image, LNCnts3, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNCnts1 = contours.sort_contours(LNCnts1, method="top-to-bottom")[0]
		LNCnts2 = contours.sort_contours(LNCnts2, method="top-to-bottom")[0]
		LNCnts3 = contours.sort_contours(LNCnts3, method="top-to-bottom")[0]
		LNCnts4 = contours.sort_contours(LNCnts4, method="top-to-bottom")[0]
		LNCnts5 = contours.sort_contours(LNCnts5, method="top-to-bottom")[0]
		LNCnts6 = contours.sort_contours(LNCnts6, method="top-to-bottom")[0]
		LNCnts7 = contours.sort_contours(LNCnts7, method="top-to-bottom")[0]
		LNCnts8 = contours.sort_contours(LNCnts8, method="top-to-bottom")[0]
		LNCnts9 = contours.sort_contours(LNCnts9, method="top-to-bottom")[0]

		LNContoursList = [LNCnts1, LNCnts2, LNCnts3, LNCnts4,LNCnts5,LNCnts6,LNCnts7,LNCnts8,LNCnts9]
		LNContoursMat = np.array(LNContoursList)
		# print(LNContoursMat)

		letter = ""
		lastname = ""
		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts1):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts2):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter


		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts3):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts4):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts5):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts6):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts7):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts8):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts9):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter

		print("UID")
		print(lastname)

		uid = lastname



		###### ADDITIONAL INFO ###############
		LETTERS = ["0", "1","2","3","4","5","6","7","8","9"]

		last_name_image = aligned_ln[445:730, 753:985]
		# cv2.imshow("Additional Info Region", last_name_image)
		# cv2.waitKey(0)

		gray_ln = cv2.cvtColor(last_name_image, cv2.COLOR_BGR2GRAY)
		thresh_ln = cv2.threshold(gray_ln, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

		#cv2.imshow("Thresh", thresh_ln)
		#cv2.waitKey(0)

		# find contours in the thresholded image, then initialize
		# the list of contours that correspond to questions
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

		#cv2.drawContours(last_name_image, lastNameCnts, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNcnts = []
		for x in range(0, len(lastNameCnts), 9):
			SortedLR = lastNameCnts[x:x+9]
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

		for x in range(0, len(questionCnts), 9):
			LNCnts1 += LNcnts[x:x + 1]
			LNCnts2 += LNcnts[x+1:x+2]
			LNCnts3 += LNcnts[x+2:x+3]
			LNCnts4 += LNcnts[x+3:x+4]
			LNCnts5 += LNcnts[x+4:x+5]
			LNCnts6 += LNcnts[x+5:x+6]
			LNCnts7 += LNcnts[x+6:x+7]
			LNCnts8 += LNcnts[x+7:x+8]
			LNCnts9 += LNcnts[x+8:x+9]



		#cv2.drawContours(last_name_image, LNCnts3, -1, (0, 0, 255), 3)
		#cv2.imshow('Bubbles', last_name_image)
		#cv2.waitKey(0)

		LNCnts1 = contours.sort_contours(LNCnts1, method="top-to-bottom")[0]
		LNCnts2 = contours.sort_contours(LNCnts2, method="top-to-bottom")[0]
		LNCnts3 = contours.sort_contours(LNCnts3, method="top-to-bottom")[0]
		LNCnts4 = contours.sort_contours(LNCnts4, method="top-to-bottom")[0]
		LNCnts5 = contours.sort_contours(LNCnts5, method="top-to-bottom")[0]
		LNCnts6 = contours.sort_contours(LNCnts6, method="top-to-bottom")[0]
		LNCnts7 = contours.sort_contours(LNCnts7, method="top-to-bottom")[0]
		LNCnts8 = contours.sort_contours(LNCnts8, method="top-to-bottom")[0]
		LNCnts9 = contours.sort_contours(LNCnts9, method="top-to-bottom")[0]

		LNContoursList = [LNCnts1, LNCnts2, LNCnts3, LNCnts4,LNCnts5,LNCnts6,LNCnts7,LNCnts8,LNCnts9]
		LNContoursMat = np.array(LNContoursList)
		# print(LNContoursMat)

		letter = ""
		lastname = ""
		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts1):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter


		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts2):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts3):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts4):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts5):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts6):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts7):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts8):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		bubbled = None
		# loop over the sorted contours
		for (j, c) in enumerate(LNCnts9):
			# construct a mask that reveals only the current
			# "bubble" for the question
			mask = np.zeros(thresh_ln.shape, dtype="uint8")
			cv2.drawContours(mask, [c], -1, 255, -1)
			# apply the mask to the thresholded image, then
			# count the number of non-zero pixels in the
			# bubble area
			mask = cv2.bitwise_and(thresh_ln, thresh_ln, mask=mask)
			# cv2.imshow('Mask', mask)
			# cv2.waitKey(0)
			total = cv2.countNonZero(mask)
			# if the current total has a larger number of total
			# non-zero pixels, then we are examining the currently
			# bubbled-in answer
			if total > T:
				if bubbled is None or total > bubbled[0]:
					bubbled = (total, j)
					letter = LETTERS[j]
					lastname += letter



		print("Additional Info")
		print(lastname)

		add = lastname




		print("")
		print("Last Name")
		print(last_name)

		print("")
		print("First Name")
		print(first_name)

		print("")
		print("UID")
		print(uid)

		print("")
		print("Additional Info")
		print(add)

		print("")
		print("Score")
		print(score)

		writer.writerow([last_name, first_name, score, uid, add])

		# print(output_np)
		# output_mat = np.vstack((output_mat, output))



