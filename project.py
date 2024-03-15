from __future__ import print_function
from __future__ import division

import time
import brickpi3
import math
import numpy as np
import cv2
from picamera2 import Picamera2, Preview

import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import pdb

BP = brickpi3.BrickPi3()

rightMotor = BP.PORT_D
leftMotor = BP.PORT_A

# constants
pi = math.pi
wheel_radius = 2.8 # cm
wheel_distance = 14.25 # cm
max_acceleration = wheel_radius * 9 * pi # cms^(-2)
max_velocity = 0.53 * max_acceleration # cms^(-1)
floor_modifier_move = 1.02 # 1.02 = hard floor, ? = carpet
floor_modifier_rotate = 1.08 # 1.08 = hard floor, ? = carpet

camera_homography_close = np.array([
	(0.078848, -0.0043416, -25.806),
	(-0.00044719, -0.044198, 48.690),
	(0.00026732, 0.0030178, 1)], dtype = float)

camera_homography_far = np.array([
	( 1.89262936e-01,  5.69764586e-02, -6.73290016e+01),
	(-3.70292742e-03, -1.68209037e-01,  1.31282103e+02),
	( 2.50165420e-05,  3.94712742e-03,  1.00000000e+00)], dtype = float)

# x1: float (cm)
# y1: float (cm)
# x2: float (cm)
# y1: float (cm)
def distance(x1, y1, x2, y2):
	return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# length: float (cm)
def moveStraight(length):
	try:
		length = length * floor_modifier_move
		backwards = length < 0
		degrees = 180 * length / (pi * wheel_radius)
		delta_time = length / 15

		BP.offset_motor_encoder(rightMotor, BP.get_motor_encoder(rightMotor))
		BP.offset_motor_encoder(leftMotor, BP.get_motor_encoder(leftMotor))
		BP.set_motor_limits(rightMotor, 70, 360)
		BP.set_motor_limits(leftMotor, 70, 360)
		current_time = time.time()
		end_time = current_time + delta_time
		BP.set_motor_position(rightMotor, degrees)
		BP.set_motor_position(leftMotor, degrees)

		while (time.time() <= end_time):
			if not backwards:
				while (BP.get_motor_encoder(rightMotor) > int(degrees + 2) and BP.get_motor_encoder(leftMotor) > int(degrees + 2)):
					continue

			else:
				while (BP.get_motor_encoder(rightMotor) < int(degrees - 2) and BP.get_motor_encoder(leftMotor) < int(degrees - 2)):
					continue
	except IOError as error:
		print(error)

# angle: float (degrees)
def rotateAntiClockwise(angle):
	try:
		angle = angle * floor_modifier_rotate
		delta_time = angle / 90
		angular_velocity = (1 / delta_time) * angle * wheel_distance / (2 * wheel_radius)

		BP.offset_motor_encoder(rightMotor, BP.get_motor_encoder(rightMotor))
		BP.offset_motor_encoder(leftMotor, BP.get_motor_encoder(leftMotor))
		BP.set_motor_limits(rightMotor, 70, 360)
		BP.set_motor_limits(leftMotor, 70, 360)

		current_time = time.time()
		end_time = current_time + delta_time
		BP.set_motor_dps(rightMotor, angular_velocity)
		BP.set_motor_dps(leftMotor, -angular_velocity)
		while (time.time() <= end_time):
			continue
		BP.set_motor_dps(rightMotor, 0)
		BP.set_motor_dps(leftMotor, 0)
		time.sleep(0.1)

	except IOError as error:
		print(error)

# x: float (cm)
# y: float (cm)
# theta: float (degrees)
# x_target: float (cm)
# y_target: float (cm)
def moveStraightToWaypoint(x, y, theta, x_target, y_target):
	alpha = math.atan2(y_target - y, x_target - x) * 180 / pi

	beta = alpha - theta
	if beta > 180:
		beta -= 360
	elif beta < -180:
		beta += 360

	rotateAntiClockwise(beta)
	moveStraight(distance(x, y, x_target, y_target))

# x: float(cm)
# y: float(cm)
# x_new: float(cm)
# y_new: float(cm)
# x_obstacle: float(cm)
# y_obstacle: float(cm)
# x_target: float (cm)
# y_target: float (cm)
# return: float (cm)
def costBenefit(x, y, x_new, y_new, x_obstacle, y_obstacle, x_target, y_target):
	weight_benefit = 12
	weight_cost = 16
	safe_distance = 18 # cm
	radius_robot = wheel_distance / 2 # cm
	radius_obstacle = 2 # cm

	benefit = weight_benefit * (distance(x, y, x_target, y_target) - distance(x_new, y_new, x_target, y_target))

	distance_to_obstacle = distance(x_new, y_new, x_obstacle, y_obstacle) - radius_robot - radius_obstacle
	if distance_to_obstacle < safe_distance:
		cost = weight_cost * (safe_distance - distance_to_obstacle)
	else:
		cost = 0

	return benefit - cost


# return: [(x_obstacle, y_obstacle)]
def getObstacles():
	# placeholder until we implement camera
	return [(20, 0), (40, -40), (40, 40), (60, 0), (80, -40), (80,40), (50, 0), (60, 30)]

# x: float (cm)
# y: float (cm)
# obstacle_list: list [(x_obstacle, y_obstacle)]
# return: (x_closest (cm), y_closest (cm))
def getClosestObstacle(x, y, obstacle_list):
	shortest_distance = 10000.0
	x_closest = 10000.0
	y_closest = 10000.0

	for obstacle in obstacle_list:
		(x_obstacle, y_obstacle) = obstacle
		obstacle_distance = distance(x, y, x_obstacle, y_obstacle)
		if obstacle_distance < shortest_distance:
			shortest_distance = obstacle_distance
			x_closest = x_obstacle
			y_closest = y_obstacle

	return (x_closest, y_closest)

# velocity_l: float (cms^-2)
# velocity_r: float (cms^-2)
# x: float (cm)
# y: float (cm)
# theta: float (radians)
# delta_time: float (s)
# return: (float(cm), float(cm), float(radians))
def predictMovement(velocity_l, velocity_r, x, y, theta, delta_time):
	if velocity_l == velocity_r:
		x_new = x + velocity_l * delta_time * math.cos(theta)
		y_new = y + velocity_l * delta_time * math.sin(theta)
		theta_new = theta

	elif velocity_l == -velocity_r:
		x_new = x
		y_new = y
		theta_new = theta + ((velocity_r - velocity_l) * delta_time / wheel_distance)

	else:
		arc_radius = wheel_distance / 2 * (velocity_l + velocity_r) / (velocity_r - velocity_l)
		delta_theta = (velocity_r - velocity_l) * delta_time / wheel_distance
		x_new = x + arc_radius * (math.sin(delta_theta + theta) - math.sin(theta))
		y_new = y - arc_radius * (math.cos(delta_theta + theta) - math.cos(theta))
		theta_new = theta + delta_theta

	return (x_new, y_new, theta_new)

def getTruePosition(x, y, theta, relative_x, relative_y):
	true_x = x + relative_y * math.cos(theta) + relative_x * math.sin(theta)
	true_y = y + relative_y * math.sin(theta) - relative_x * math.cos(theta)

	return (true_x, true_y)

def dynamicWindowApproach():
	try:
		picam2 = Picamera2()
		preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
#		preview_config = picam2.create_preview_configuration()

		picam2.configure(preview_config)
		picam2.start()
		white = (255,255,255)

		x_start = 0.0
		y_start = 0.0
		theta_start = 0.0
		velocity_l_start = 0.0
		velocity_r_start = 0.0
		delta_time = 0.08
		obstacles = []
		starttime = time.time()

		x_target, y_target = (150, 0) # We identify static target

		while True:
			#obstacles = []
			best_cost_benefit = -10000.0
			start_calc = time.time()

			img = picam2.capture_array()
			print("\n Captured image at time", start_calc - starttime)

			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

			# lower mask (0-10)
			lower_red = np.array([0,50,50])
			upper_red = np.array([10,255,255])
			mask0 = cv2.inRange(hsv, lower_red, upper_red)

			# upper mask (170-180)
			lower_red = np.array([170,50,50])
			upper_red = np.array([180,255,255])
			mask1 = cv2.inRange(hsv, lower_red, upper_red)

			# join my masks
			mask = mask0+mask1
			result = cv2.bitwise_and(img, img, mask=mask)

			output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32F)
			(numLabels, labels, stats, centroids) = output

			# loop over the number of unique connected component labels
			for i in range(0, numLabels):
				# i=0 is the background region
				if i != 0:
					# extract the connected component statistics and centroid
					x = stats[i, cv2.CC_STAT_LEFT]
					y = stats[i, cv2.CC_STAT_TOP]
					w = stats[i, cv2.CC_STAT_WIDTH]
					h = stats[i, cv2.CC_STAT_HEIGHT]
					area = stats[i, cv2.CC_STAT_AREA]
					(cX, cY) = centroids[i]
					if (area > 50):
						#print("Component", i, "area", area, "Centroid", cX, cY)
						#img = cv2.circle(img, (int(cX), int(cY)), 5, white, 3)
						adjustedPixels = np.array([cX, cY, 1])
						relativePosition = predictCoordinates(adjustedPixels)
						relative_x = relativePosition[0]
						relative_y = relativePosition[1]
						(obstacle_x, obstacle_y) = getTruePosition(x_start, y_start, theta_start, relative_x, relative_y)
						if any(distance(obst[0], obst[1], obstacle_x, obstacle_y) <= 15 for obst in obstacles):
							continue
						obstacles.append((obstacle_x, obstacle_y))

			print(obstacles)
			end_calc = time.time()
			calc_time = end_calc - start_calc
			possible_velocities_l = [velocity_l_start - max_acceleration * delta_time, velocity_l_start, velocity_l_start + max_acceleration * delta_time]
			possible_velocities_r = [velocity_r_start - max_acceleration * delta_time, velocity_r_start, velocity_r_start + max_acceleration * delta_time]

			for velocity_l in possible_velocities_l:
				for velocity_r in possible_velocities_r:
					if velocity_l <= max_velocity and velocity_r <= max_velocity and velocity_l >= -max_velocity and velocity_r >= -max_velocity:
						(x_new, y_new, theta_new) = predictMovement(velocity_l, velocity_r, x_start, y_start, theta_start, 10 * delta_time)
						(x_obstacle, y_obstacle) = getClosestObstacle(x_new, y_new, obstacles)
						cost_benefit = costBenefit(x_start, y_start, x_new, y_new, x_obstacle, y_obstacle, x_target, y_target)
						if cost_benefit > best_cost_benefit:
							velocity_l_chosen = velocity_l
							velocity_r_chosen = velocity_r
							best_cost_benefit = cost_benefit

			BP.set_motor_dps(leftMotor, (velocity_l_chosen / wheel_radius) * (180 / pi))
			BP.set_motor_dps(rightMotor, (velocity_r_chosen / wheel_radius) * (180 / pi))
			print("vl: " + str(velocity_l_chosen) + ", vr: " + str(velocity_r_chosen))
			velocity_l_start = velocity_l_chosen
			velocity_r_start = velocity_r_chosen
			print("current position: (x" + str(x_start) + ", y" + str(y_start) + ", theta" + str(theta_start) + ")")
			x_start, y_start, theta_start = predictMovement(velocity_l_start, velocity_r_start, x_start, y_start, theta_start, delta_time)
			end_calc = time.time()
			calc_time = end_calc - start_calc
			print("Calculation time: ", calc_time)
			if (start_calc - starttime > 0.3):
				if delta_time > calc_time:
					time.sleep(delta_time - calc_time)
			else:
				time.sleep(delta_time)

	except KeyboardInterrupt:
		BP.reset_all()
		picam2.stop()
		cv2.destroyAllWindows()

# pixels: numpy (3x1) array [pix_x, pix_y, 1]
# return: numpy (3x1) array [coord_x, coord_y, 1]
def predictCoordinates(pixels):
	estimate = np.dot(camera_homography_far, pixels)
	estimate = estimate / estimate[2]
	return estimate

def enableCamera():
	picam2 = Picamera2()
#	preview_config = picam2.create_preview_configuration(main={"size": (1152, 648)})
#	preview_config = picam2.create_preview_configuration(main={"size": (2304, 1296)})
	preview_config = picam2.create_preview_configuration()
	picam2.configure(preview_config)

	picam2.start()

	starttime = time.time()

	white = (255,255,255)

#	for j in range(1000):
	while True:
		img = picam2.capture_array()

		# Applying 7x7 Gaussian Blur
		#img = cv2.GaussianBlur(img, (27, 27), 0)

		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

		# lower mask (0-10)
		lower_red = np.array([0,50,50])
		upper_red = np.array([10,255,255])
		mask0 = cv2.inRange(hsv, lower_red, upper_red)

		# upper mask (170-180)
		lower_red = np.array([170,50,50])
		upper_red = np.array([180,255,255])
		mask1 = cv2.inRange(hsv, lower_red, upper_red)

		# join my masks
		mask = mask0+mask1
		result = cv2.bitwise_and(img, img, mask=mask)

		output = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32F)
		(numLabels, labels, stats, centroids) = output

		# loop over the number of unique connected component labels
		for i in range(0, numLabels):
			# i=0 is the background region
			if i != 0:
				# extract the connected component statistics and centroid
				x = stats[i, cv2.CC_STAT_LEFT]
				y = stats[i, cv2.CC_STAT_TOP]
				w = stats[i, cv2.CC_STAT_WIDTH]
				h = stats[i, cv2.CC_STAT_HEIGHT]
				area = stats[i, cv2.CC_STAT_AREA]
				(cX, cY) = centroids[i]
				if (area > 35):
					print("Component", i, "area", area, "Centroid", cX, cY)
					img = cv2.circle(img, (int(cX), int(cY)), 5, white, 3)

		#font = cv2.FONT_HERSHEY_SIMPLEX

		cv2.imwrite("Photos/demo.jpg", img)
		#print("drawImg:" + "/home/pi/RoboticsFYP/Photos/demo.jpg")
#		print("Captured image", j, "at time", time.time() - starttime)
		print("\n")
		cv2.imshow("Camera", img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	picam2.stop()
	cv2.destroyAllWindows()

def colourTest():
	picam2 = Picamera2()
	preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
	picam2.configure(preview_config)

	picam2.start()

	starttime = time.time()

	white = (255,255,255)

	while True:
		frame = picam2.capture_array()
		frameNotBlue = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		frameHSV = cv2.cvtColor(frameNotBlue, cv2.COLOR_BGR2HSV)
		circled = cv2.circle(frame, (320, 240), 7, white, 1)

		print(frameHSV[320, 240])

		cv2.imshow("Camera", frameNotBlue)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	picam2.stop()
	cv2.destroyAllWindows()

def calculateHomography():
	(x1, y1, u1, v1) = (20, 40, 587, 220)
	(x2, y2, u2, v2) = (-20, 40, 38, 229)
	(x3, y3, u3, v3) = (10, 20, 566, 431)
	(x4, y4, u4, v4) = (-10, 20, 72, 436)

	A = np.array([[x1, y1, 1, 0, 0, 0, -u1 * x1, -u1 * y1],
	              [0, 0, 0, x1, y1, 1, -v1 * x1, -v1 * y1],
	              [x2, y2, 1, 0, 0, 0, -u2 * x2, -u2 * y2],
	              [0, 0, 0, x2, y2, 1, -v2 * x2, -v2 * y2],
	              [x3, y3, 1, 0, 0, 0, -u3 * x3, -u3 * y3],
	              [0, 0, 0, x3, y3, 1, -v3 * x3, -v3 * y3],
	              [x4, y4, 1, 0, 0, 0, -u4 * x4, -u4 * y4],
	              [0, 0, 0, x4, y4, 1, -v4 * x4, -v4 * y4]])

	b = np.array([u1, v1, u2, v2, u3, v3, u4, v4])

	R, residuals, RANK, sing = np.linalg.lstsq(A, b, rcond=None)

	H = np.array([[R[0], R[1], R[2]],
	              [R[3], R[4], R[5]],
	              [R[6], R[7], 1]])

	HInv = np.linalg.inv(H)
	HInv = HInv / HInv[2][2]

	print ("Homography")
	print (HInv)



dynamicWindowApproach()
#BP.reset_all()

"""
try:
	#print("All good!")
	calculateHomography()
#	enableCamera()
#	colourTest()
#	dynamicWindowApproach()
	#while True:
	#	BP.set_motor_dps(rightMotor, 180)
	#	BP.set_motor_dps(leftMotor, 180)
	#print(homographyError(depth_60))
	#drawContourMap()
	#BP.reset_all()
except KeyboardInterrupt:
	BP.reset_all()

#"""
