from __future__ import print_function
from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import brickpi3
import math
import numpy as np
import random

BP = brickpi3.BrickPi3()

rightMotor = BP.PORT_D
leftMotor = BP.PORT_A

# constants
pi = math.pi
wheel_radius = 2.8 # cm
wheel_distance = 14.25 # cm
floor_modifier_move = 1.02 # 1.02 = hard floor, ? = carpet
floor_modifier_rotate = 1.08 # 1.08 = hard floor, ? = carpet
camera_homography = np.array([
	(0.025307, 0.000016202, -56.024),
	(-0.00037511, -0.014528, 65.596),
	(0.000019383, 0.00080886, 1)], dtype = float)

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
def costBenefit(x, y, x_new, y_new, x_target, y_target):
	weight_benefit = 1.0
	weight_cost = 1.0
	safe_distance = 5 # cm
	radius_robot = 8 # cm
	radius_obstacle = 3.5 # cm

	benefit = weight_benefit * (distance(x, y, x_target, y_target) - distance(x_new, y_new, x_target, y_target))
	cost = weight_cost * (safe_distance - distance(x_new, y_new, x_obstacle, y_obstacle) - radius_robot - radius_obstacle)
	return benefit - cost

# x: float (cm)
# y: float (cm)
# obstacle_list: list [(x_obstacle, y_obstacle)]
def closestObstacle(x, y, obstacle_list):
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

# velocity_l: float (cms^-2)
# velocity_r: float (cms^-2)
# x: float (cm)
# y: float (cm)
# theta: float (degrees)
# delta_time: float (s)
# return: (float(cm), float(cm), float(degrees))
def predictMovement(velocity_l, velocity_r, x, y, theta, delta_time):
	if velocity_l == velocity_r:
		x_new = x + velocity_l * delta_time * math.cos(theta * pi / 180)
		y_new = y + velocity_l * delta_time * math.sin(theta * pi / 180)
		theta_new = theta

	elif velocity_l == -velocity_r:
		x_new = x
		y_new = y
		theta_new = theta + ((velocity_r - velocity_l) * delta_time / wheel_distance)

	else:
		arc_radius = wheel_distance / 2 * (velocity_l + velocity_r) / (velocity_r - velocity_l)
		delta_theta = (velocity_r - velocity_l) * delta_time / wheel_distance
		x_new = x + arc_radius * (math.sin((delta_theta + theta) * pi / 180) - math.sin(theta * pi / 180))
		y_new = y + arc_radius * (math.cos((delta_theta + theta) * pi / 180) - math.cos(theta * pi / 180))
		theta_new = theta + delta_theta

	return (x_new, y_new, theta_new)

# pixels: numpy (3x1) array [pix_x, pix_y, 1]
# return: numpy (3x1) array [coord_x, coord_y, 1]
def predictCoordinates(pixels):
	estimate = np.dot(camera_homography, pixels)
	estimate = estimate / estimate[2]
	return estimate

# row: numpy array [(pix_x, pix_y, coord_x, coord_y)]
# return: numpy array of errors in cm, at pixel coords
# 	: [[pix_x, pix_y, error]]
def homographyError(row):
	nPoints = row.shape[0]
	outputErrors = np.empty((nPoints, 3))
	for i in range(nPoints):
		pix_x = row[i][0]
		pix_y = row[i][1]
		coord_x = row[i][2]
		coord_y = row[i][3]
		pixels = np.array([pix_x, pix_y, 1])

		estimate = predictCoordinates(pixels)
		error_x = abs(coord_x - estimate[0])
		error_y = abs(coord_y - estimate[1])
		error = math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2))
		outputErrors[i][0] = pix_x
		outputErrors[i][1] = pix_y
		outputErrors[i][2] = error

	return outputErrors

def drawContourMap():
	depth_far = np.array([
		(190, 48, -60, 70),
		(2621, 30, 10, 62),
		(4240, 11, 60, 70),
		(4460, 83, 70, 70)])

	depth_60 = np.array([
		(16, 218, -60, 60),
		(250, 142, -50, 60),
		(561, 80, -40, 60),
		(946, 54, -30, 60),
		(1375, 55, -20, 60),
		(1798, 56, -10, 60),
		(2215, 57, 0, 60),
		(2631, 62, 10, 60),
		(3040, 59, 20, 60),
		(3453, 59, 30, 60),
		(3843, 73, 40, 60),
		(4178, 115, 50, 60),
		(4437, 179, 60, 60)])

	depth_50 = np.array([
		(30, 362, -50, 50),
		(334, 289, -40, 50),
		(754, 250, -30, 50),
		(1249, 250, -20, 50),
		(1736, 252, -10, 50),
		(2211, 249, 0, 50),
		(2685, 250, 10, 50),
		(3153, 251, 20, 50),
		(3627, 248, 30, 50),
		(4064, 266, 40, 50),
		(4407, 320, 50, 50)])

	depth_40 = np.array([
		(77, 570, -40, 40),
		(504, 519, -30, 40),
		(1077, 510, -20, 40),
		(1649, 510, -10, 40),
		(2203, 504, 0, 40),
		(2758, 509, 10, 40),
		(3306, 509, 20, 40),
		(3863, 504, 30, 40),
		(4335, 529, 40, 40),
		(4586, 571, 48, 40)])

	depth_30 = np.array([
		(47, 917, -33, 30),
		(186, 903, -30, 30),
		(824, 882, -20, 30),
		(1528, 883, -10, 30),
		(2211, 864, 0, 30),
		(2890, 860, 10, 30),
		(3546, 857, 20, 30),
		(4200, 865, 30, 30),
		(4581, 890, 38, 30)])

	depth_20 = np.array([
		(90, 1465, -25, 20),
		(453, 1464, -20, 20),
		(1343, 1449, -10, 20),
		(2214, 1428, 0, 20),
		(3075, 1408, 10, 20),
		(3900, 1396, 20, 20),
		(4588, 1405, 30, 20)])

	depth_15 = np.array([
		(28, 1856, -23, 15),
		(230, 1875, -20, 15),
		(689, 1879, -15, 15),
		(1199, 1856, -10, 15),
		(2214, 1840, 0, 15),
		(3199, 1816, 10, 15),
		(4155, 1784, 20, 15),
		(4516, 1756, 25, 15)])

	depth_10 = np.array([
		(68, 2345, -20, 10),
		(464, 2422, -15, 10),
		(1017, 2436, -10, 10),
		(1615, 2410, -5, 10),
		(2216, 2398, 0, 10),
		(2808, 2373, 5, 10),
		(3388, 2357, 10, 10),
		(3943, 2336, 15, 10),
		(4377, 2271, 20, 10),
		(4580, 2211, 23, 10)])

	depth_close = np.array([
		(35, 2538, -20, 8),
		(160, 2586, -18, 8),
		(427, 2541, -15, 9),
		(973, 2579, -10, 9),
		(1591, 2554, -5, 9),
		(2213, 2540, 0, 9),
		(2831, 2513, 5, 9),
		(3439, 2491, 10, 9),
		(4046, 2577, 15, 8),
		(4467, 2568, 20, 7),
		(4588, 2510, 22, 7),
		(4594, 2554, 22, 6.5)])

	all_depths = np.vstack((
		depth_close,
		depth_10,
		depth_15,
		depth_20,
		depth_30,
		depth_40,
		depth_50,
		depth_60,
		depth_far))

	mpl.rcParams["font.size"] = 14
	mpl.rcParams["legend.fontsize"] = "large"
	mpl.rcParams["figure.titlesize"] = "medium"
	fig, ax = plt.subplots()
	ax.xaxis.tick_top()

	errorArray = homographyError(all_depths)
	x = errorArray[:,0]
	y = errorArray[:,1]
	error = errorArray[:,2]

	ax.tricontour(x, y, error, levels=54, linewidths=0.1, colors="k")
	cntr = ax.tricontourf(x, y, error, levels=54, cmap="RdBu_r")
	cbar = fig.colorbar(cntr, ax=ax)
	cbar.set_label("error / cm", rotation = 0, labelpad = 40)
	ax.plot(x, y, "ko", ms=1.5)
	ax.set(xlim=(0, 4608), ylim=(0, 2592))
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("x / pixels", labelpad = 15)
	ax.set_ylabel("y / pixels", rotation = 0, labelpad = 15)
	ax.xaxis.set_label_position("top")

	plt.subplots_adjust(hspace=0.5)
	plt.gca().invert_yaxis()
	plt.savefig("Photos/contour_map.png")
	plt.show()

try:
	print("Everything compiles!")
	#print(homographyError(depth_60))
	#drawContourMap()
	#moveStraight(40)
	#rotateAntiClockwise(90)
	#moveStraight(40)
	#rotateAntiClockwise(90)
	#moveStraight(40)
	#rotateAntiClockwise(90)
	#moveStraight(40)
	#rotateAntiClockwise(90)
	#BP.reset_all()
except KeyboardInterrupt:
	BP.reset_all()

