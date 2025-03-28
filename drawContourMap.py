# row: numpy array [(pix_x, pix_y, coord_x, coord_y)]
# return: numpy array of errors in cm, at pixel coords
#       : [[pix_x, pix_y, error]]
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
