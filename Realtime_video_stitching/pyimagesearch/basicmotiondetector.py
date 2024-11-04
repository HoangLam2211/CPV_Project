# import the necessary packages
import imutils
import cv2

class BasicMotionDetector:
	def __init__(self, accumWeight=0.5, deltaThresh=5, minArea=5000):
		# determine the OpenCV version, followed by storing the
		# the frame accumulation weight, the fixed threshold for
		# the delta image, and finally the minimum area required
		# for "motion" to be reported
		self.isv2 = imutils.is_cv2()
		self.accumWeight = accumWeight
		self.deltaThresh = deltaThresh
		self.minArea = minArea

		# initialize the average image for motion detection
		self.avg = None

	def update(self, image):
		locs = []

		if self.avg is None:
			self.avg = image.astype("float")
			return locs

		cv2.accumulateWeighted(image, self.avg, self.accumWeight)
		frameDelta = cv2.absdiff(image, cv2.convertScaleAbs(self.avg))
		thresh = cv2.threshold(frameDelta, self.deltaThresh, 255, cv2.THRESH_BINARY)[1]
		thresh = cv2.dilate(thresh, None, iterations=2)

		# Find contours
		cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if self.isv2 else cnts[1]

		print(f"Contours found: {len(cnts) if cnts is not None else 0}")

		if cnts is not None:
			for i, c in enumerate(cnts):
				# Check if contour is valid and has enough points
				if c is not None and len(c) >= 3:
					# Print details about the contour
					print(f"Contour {i}: length={len(c)}, points={c.shape if len(c.shape) > 0 else 'N/A'}")

					area = cv2.contourArea(c)
					print(f"Contour {i} area: {area}")

					# Proceed if the area is greater than the minimum area
					if area > self.minArea:
						locs.append(c)
				else:
					print(f"Contour {i} is invalid or does not have enough points")

		return locs

