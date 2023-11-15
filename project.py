from __future__ import print_function
from __future__ import division

import time
import brickpi3
import math
import numpy as np
import random

BP = brickpi3.BrickPi3()

rightMotor = BP.PORT_C
leftMotor = BP.PORT_B

# constants
pi = math.pi
wheel_radius = 2.8 # cm
floor_modifier_move = 1.0 # 1.0 = hard floor, ? = carpet
floor_modifier_rotate = 1.0 # 1.0 = hard floor, ? = carpet

# length: cm
def moveStraight(length):
	try:
		length = length * floor_modifier_move
		backwards = length < 0
		degrees = 180 * length / (pi * wheel_radius)

		BP.offset_motor_encoder(rightMotor, BP.get_motor_encoder(rightMotor))
		BP.offset_motor_encoder(leftMotor, BP.get_motor_encoder(leftMotor))
		BP.set_motor_limits(rightMotor, 70, 300)
		BP.set_motor_limits(leftMotor, 70, 300)
		BP.set_motor_position(rightMotor, degrees)
		BP.set_motor_position(leftMotor, degrees)

		if not backwards:
			while (BP.get_motor_encoder(rightMotor) > int(degrees + 2) and BP.get_motor_encoder(leftMotor) > int(degrees + 2)):
				continue

		else:
			while (BP.get_motor_encoder(rightMotor) < int(degrees - 2) and BP.get_motor_encoder(leftMotor) < int(degrees - 2)):
				continue
	except IOError as error:
		print(error)



try:
	moveStraight(30)
	#BP.reset_all()
except KeyboardInterrupt:
	BP.reset_all()

