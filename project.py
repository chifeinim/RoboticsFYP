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

# length: cm
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

# angle: degrees
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

try:
	moveStraight(40)
	rotateAntiClockwise(90)
	moveStraight(40)
	rotateAntiClockwise(90)
	moveStraight(40)
	rotateAntiClockwise(90)
	moveStraight(40)
	rotateAntiClockwise(90)
	#BP.reset_all()
except KeyboardInterrupt:
	BP.reset_all()

