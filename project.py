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
max_velocity = 0.45 * max_acceleration # cms^(-1)
floor_modifier_move = 1.02 # 1.02 = hard floor, ? = carpet
floor_modifier_rotate = 1.0 # 1.1 = hard floor, ? = carpet

#make new homography for far, mid, close, use when appropriate. either 3 or 9.
camera_homography_far = np.array([
	( 7.66809785e-01, -2.85697817e-02, -2.33552530e+02),
	(-9.08274197e-03, -2.99672902e-01,  4.58335936e+02),
	( 1.84507928e-04,  4.16011207e-02,  1.00000000e+00)], dtype = float)

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
	true_x = x + relative_x * math.cos(theta) - relative_y * math.sin(theta)
	true_y = y + relative_x * math.sin(theta) + relative_y * math.cos(theta)

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
		delta_time = 0.1
		obstacles = []
		starttime = time.time()

		x_target, y_target = (130, 0) # We identify static target

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
			print("current position: (x" + str(x_start) + ", y" + str(y_start) + ", theta" + str(theta_start * 180 / pi) + ")")
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
# return: numpy (2x1) array [coord_x, coord_y] in conventional robot frame
def predictCoordinates(pixels):
	estimate = np.dot(camera_homography_far, pixels)
	#Currently the coordinates are in standard cartesian with robot facing positive y when theta=0.
	estimate = estimate / estimate[2]
	#We convert into conventional robot frame, with robot facing positive x when theta=0.
	conventional_x =  estimate[1]
	conventional_y = -estimate[0]
	return np.array([conventional_x, conventional_y])

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
				if (area > 75):
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
	(x1, y1, u1, v1) = (28, 70, 527.2, 117.3)
	(x2, y2, u2, v2) = (-28, 70, 89.2, 120.3)
	(x3, y3, u3, v3) = (12, 18, 606.1, 412.8)
	(x4, y4, u4, v4) = (-12, 18, 31.3, 419.6)

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

########## MCL BEGINS ##########
################################
################################

# A Canvas class for drawing a map and particles:
#     - it takes care of a proper scaling and coordinate transformation between
#      the map frame of reference (in cm) and the display (in pixels)
class Canvas:
	def __init__(self,map_size=80):
		self.map_size    = map_size    # in cm;
		self.canvas_size = 768;         # in pixels;
		self.margin      = 0.25*map_size
		self.scale       = self.canvas_size/(map_size+2*self.margin)

	def drawLine(self,line):
		x1 = self.__screenX(line[0])
		y1 = self.__screenY(line[1])
		x2 = self.__screenX(line[2])
		y2 = self.__screenY(line[3])
		print("drawLine:" + str((x1,y1,x2,y2)))

	def drawParticles(self,data):
		display = [(self.__screenX(d[0]),self.__screenY(d[1])) + d[2:] for d in data]
		print("drawParticles:" + str(display))

	def __screenX(self,x):
		return (x + self.margin)*self.scale

	def __screenY(self,y):
		return (self.map_size + self.margin - y)*self.scale

# A Map class containing walls
class Map:
	def __init__(self):
		self.walls = []

	def add_wall(self,wall):
		self.walls.append(wall)

	def clear(self):
		self.walls = []

	def draw(self, canvas):
		for wall in self.walls:
			canvas.drawLine(wall)

# Simple Particles set
class Particles:
	def __init__(self, N=100):
		self.n = N
		self.data = np.zeros((N,4))
		self.data[:,3] = 1 / N

	def initialise(self, x, y, theta):
		self.data = np.zeros((self.n, 4))

		ex = np.random.normal(0, 0.2, self.n)
		ey = np.random.normal(0, 0.2, self.n)
		f = np.random.normal(0, 0.2, self.n)

		for k in range(self.n):
			self.data[k][0] = x + ex[k]
			self.data[k][1] = y + ey[k]
			self.data[k][2] = theta + f[k]

		#self.data[:, 0] = x
		#self.data[:, 1] = y
		#self.data[:, 2] = theta
		self.data[:, 3] = 1 / self.n

	def move(self, velocity_l, velocity_r, delta_time):
		if abs(velocity_l - velocity_r) < 0.000001:
			straight_sd = 0.01 # for every 1cm
			straight_angle_sd = 0.002 # for every 1cm
			e = np.random.normal(0, abs(straight_sd * delta_time * velocity_l), self.n)
			f = np.random.normal(0, abs(straight_angle_sd * delta_time * velocity_l), self.n)
			for k in range(self.n):
				self.data[k][0] += (velocity_l * delta_time + e[k]) * math.cos(self.data[k][2])
				self.data[k][1] += (velocity_l * delta_time + e[k]) * math.sin(self.data[k][2])
				self.data[k][2] += f[k]

		elif abs(velocity_l + velocity_r) < 0.000001:
			rotation_angle_sd = 1/90 * (pi / 180) # for every 1 radian, or 1/90 degrees per degree
			g = np.random.normal(0, rotation_angle_sd * abs(((velocity_r - velocity_l) * delta_time / wheel_distance)), self.n)
			for k in range(self.n):
				self.data[k][2] += ((velocity_r - velocity_l * delta_time / wheel_distance) + g[k])

		else:
			arc_radius_sd = 0.01 # for every 1cm
			arc_radius = wheel_distance / 2 * (velocity_l + velocity_r) / (velocity_r - velocity_l)
			delta_theta_sd = 0.004 * pi / 180 # for every 1 radian
			delta_theta = (velocity_r - velocity_l) * delta_time / wheel_distance
			h = np.random.normal(0, abs(arc_radius_sd * arc_radius), self.n)
			i = np.random.normal(0, abs(delta_theta_sd * delta_theta), self.n)
			for k in range(self.n):
				self.data[k][0] += (arc_radius + h[k]) * (math.sin(delta_theta + i[k] + self.data[k][2]) - math.sin(self.data[k][2]))
				self.data[k][1] -= (arc_radius + h[k]) * (math.cos(delta_theta + i[k] + self.data[k][2]) - math.cos(self.data[k][2]))
				self.data[k][2] += delta_theta + i[k]

	def update_weights(self, z):
		# adjust weights
		total_weight = 0
		for idx, (x, y, theta, w) in enumerate(self.data):
			prob = calculate_likelihood(x, y, theta, z)
			#print(f"Particle: %s" % ((x,y,theta),))
			#print(prob)
			self.data[idx][3] = prob * w
			total_weight += self.data[idx][3]
		self.data[:,3] /= total_weight

	def resample(self):
		cumulative_weights = self.data[:,3]
		total = 0
		new_particles = np.zeros((self.n, 4))
		for i in range(self.n):
			total += cumulative_weights[i]
			cumulative_weights[i] = total
		for i in range(self.n):
			rand = random.random()
			idx = 0
			while cumulative_weights[idx] < rand and idx < self.n - 1:
				idx += 1
			new_particle = self.data[idx]
			new_particle[3] = 1 / self.n
			new_particles[i] = new_particle

		self.data = new_particles

	def mean(self):
		return np.mean(self.data, axis=0)[:3]

	def draw(self, canvas):
		tupleParticles = list(map(tuple, self.data.reshape((self.n, 4))))
		canvas.drawParticles(tupleParticles)

def calculate_likelihood(x, y, theta, z):
	K = 0.000001
	best_likelihood = 0
	if not z or waymark_list.size:
		best_likelihood = K
	else:
		for (z_x, z_y) in z:
			(z_world_x, z_world_y) = getTruePosition(x, y, theta, z_x, z_y)
			for (x_waymark, y_waymark) in waymark_list:
				difference = distance(z_world_x, z_world_y, x_waymark, y_waymark)
				measured_distance = distance(x, y, z_world_x, z_world_y)
				sd = 0.1 * measured_distance
				likelihood = (math.e ** (-(difference ** 2) / (2 * sd ** 2))) + K
				if likelihood > best_likelihood:
					best_likelihood = likelihood
	return best_likelihood

def monteCarloLocalisation(waypoints, particles, canvas):
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
		delta_time = 0.1
		obstacles = []
		starttime = time.time()

		for (x_target, y_target) in waypoints:

			while True:
				print("Target: (" + str(x_target) + ", " + str(y_target) + ")")
				distance_to_waypoint = distance(x_start, y_start, x_target, y_target)
				print("Distance to waypoint: " + str(distance_to_waypoint))
				if distance_to_waypoint < 2:
					break

				z = []
				avg_waymarks = []
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
						if (area > 125):
							#print("Component", i, "area", area, "Centroid", cX, cY)
							#img = cv2.circle(img, (int(cX), int(cY)), 5, white, 3)
							adjustedPixels = np.array([cX, cY, 1])
							relativePosition = predictCoordinates(adjustedPixels)
							relative_x = relativePosition[0]
							relative_y = relativePosition[1]
							z.append((relative_x, relative_y))
							(avg_waymark_x, avg_waymark_y) = getTruePosition(x_start, y_start, theta_start, relative_x, relative_y)
							avg_waymarks.append((avg_waymark_x, avg_waymark_y))

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

				# We do this to make turning tighter. Has no effect on particle calculations.
				difference = abs(velocity_l_chosen - velocity_r_chosen)
				sum = abs(velocity_l_chosen + velocity_r_chosen)
				if (difference < 0.000001 or sum < 0.000001):
					BP.set_motor_dps(leftMotor, (velocity_l_chosen / wheel_radius) * (180 / pi))
					BP.set_motor_dps(rightMotor, (velocity_r_chosen / wheel_radius) * (180 / pi))
				else:
					BP.set_motor_dps(leftMotor, floor_modifier_rotate * (velocity_l_chosen / wheel_radius) * (180 / pi))
					BP.set_motor_dps(rightMotor, floor_modifier_rotate * (velocity_r_chosen / wheel_radius) * (180 / pi))

				print("vl: " + str(velocity_l_chosen) + ", vr: " + str(velocity_r_chosen))
				# New Particles code:
				particles.move(velocity_l_chosen, velocity_r_chosen, delta_time)
				#This z is purely symbolic, we actually have different z per particle
				if avg_waymarks:
					print("AVERAGE Z DETECTED: ------------------------------------> " + str(avg_waymarks))
				particles.update_weights(z)
				print("mean")
				particles.resample()
				print(particles.mean())
				particles.draw(canvas)

				velocity_l_start = velocity_l_chosen
				velocity_r_start = velocity_r_chosen
				x_start, y_start, theta_start = particles.mean()
				"""
				cv2.imshow("Camera", img)

				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				"""
				end_calc = time.time()
				calc_time = end_calc - start_calc
				print("Calculation time: ", calc_time)
				if (start_calc - starttime > 0.3):
					if delta_time > calc_time:
						time.sleep(delta_time - calc_time)
				else:
					time.sleep(delta_time)
		BP.reset_all()

	except KeyboardInterrupt:
		BP.reset_all()
		picam2.stop()
		cv2.destroyAllWindows()

def initialiseMCL(waymarks, waypoints):

	canvas = Canvas()
	my_map = Map()

	for (x, y) in waymarks:
		my_map.add_wall((x-1, y+1, x-1, y-1))
		my_map.add_wall((x-1, y-1, x+1, y-1))
		my_map.add_wall((x+1, y-1, x+1, y+1))
		my_map.add_wall((x+1, y+1, x-1, y+1))

	for (x, y) in waypoints:
		my_map.add_wall((x-1, y+1, x+1, y-1))
		my_map.add_wall((x+1, y+1, x-1, y-1))

	my_map.draw(canvas)
	particles = Particles()
	monteCarloLocalisation(waypoints, particles, canvas)

waymark_list = np.array([(56, 23), (93, 82), (-37, 36), (130, -13), (0, 82)])
#waymark_list = np.array([])
waypoint_list = np.array([(80, 0), (90, 50), (0, 50), (0, 0)])
#waypoint_list = np.array([(0, 50), (90, 50), (90, 0), (0, 0)])

initialiseMCL(waymark_list, waypoint_list)
#dynamicWindowApproach()
#enableCamera()
#calculateHomography()
BP.reset_all()
