from __future__ import print_function
from __future__ import division

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

# length: float (cm)
def moveStraight(length):
	try:
		length = length * floor_modifier_move
		backwards = length < 0
		degrees = 180 * length / (pi * wheel_radius)
		time_delta = length / 15

		BP.offset_motor_encoder(rightMotor, BP.get_motor_encoder(rightMotor))
		BP.offset_motor_encoder(leftMotor, BP.get_motor_encoder(leftMotor))
		BP.set_motor_limits(rightMotor, 70, 360)
		BP.set_motor_limits(leftMotor, 70, 360)
		current_time = time.time()
		end_time = current_time + time_delta
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
		time_delta = angle / 90
		angular_velocity = (1 / time_delta) * angle * wheel_distance / (2 * wheel_radius)

		BP.offset_motor_encoder(rightMotor, BP.get_motor_encoder(rightMotor))
		BP.offset_motor_encoder(leftMotor, BP.get_motor_encoder(leftMotor))
		BP.set_motor_limits(rightMotor, 70, 360)
		BP.set_motor_limits(leftMotor, 70, 360)

		current_time = time.time()
		end_time = current_time + time_delta
		BP.set_motor_dps(rightMotor, angular_velocity)
		BP.set_motor_dps(leftMotor, -angular_velocity)
		while (time.time() <= end_time):
			continue
		BP.set_motor_dps(rightMotor, 0)
		BP.set_motor_dps(leftMotor, 0)
		time.sleep(0.1)

	except IOError as error:
		print(error)

# pixels: numpy (3x1) array [pix_x, pix_y, 1]
# return: numpy (3x1) array [coord_x, coord_y, 1]
def predictPosition(pixels):
	estimate = np.dot(camera_homography, pixels)
	estimate = estimate / estimate[2]
	return estimate

# row: numpy array [(pix_x, pix_y, coord_x, coord_y)]
# return: numpy array of errors in cm
def homographyError(row):
	nPoints = row.shape[0]
	outputErrors = np.empty((nPoints, 1))
	for i in range(nPoints):
		pix_x = row[i][0]
		pix_y = row[i][1]
		coord_x = row[i][2]
		coord_y = row[i][3]
		pixels = np.array([pix_x, pix_y, 1])

		estimate = predictPosition(pixels)
		error_x = abs(coord_x - estimate[0])
		error_y = abs(coord_y - estimate[1])
		error = math.sqrt(math.pow(error_x, 2) + math.pow(error_y, 2))
		outputErrors[i] = error

	return outputErrors

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
	(4335, 529, 40, 40)])

depth_30 = np.array([
	(186, 903, -30, 30),
	(824, 882, -20, 30),
	(1528, 883, -10, 30),
	(2211, 864, 0, 30),
	(2890, 860, 10, 30),
	(3546, 857, 20, 30),
	(4200, 865, 30, 30)])

depth_20 = np.array([
	(90, 1465, -25, 20),
	(453, 1464, -20, 20),
	(1343, 1449, -10, 20),
	(2214, 1428, 0, 20),
	(3075, 1408, 10, 20),
	(3900, 1396, 20, 20),
	(4588, 1405, 30, 20)])

depth_15 = np.array([
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
	(4377, 2271, 20, 10)])


try:
	#a = np.array([16, 218, 1])
	#b = np.dot(camera_homography, a)
	#b = b / b[2]
	#print(b)
	print(homographyError(depth_30))
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

