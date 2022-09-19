import heapq
import math
import sys
from PIL import Image
import numpy as np

"""
return hex color code for the given RGB code.
"""


def rgb2hex(r, g, b):
    return '#%02X%02X%02X' % (r, g, b)


"""
Return 2D array of the image.
"""


def loadImage(filename):
    img = Image.open(filename).convert("RGB")
    arr = np.array(img)
    return arr


"""
Read the points to be visited
"""


def readPoints(filename):
    points = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            data = line.split(" ")
            points.append((int(data[1]), int(data[0])))
    return points


"""
Read elevations file, store the data in the array and return the array
"""


def readElevations(filename):
    with open(filename) as f:
        data = []
        for line in f:
            line = line.strip()
            line = line.replace("   "," ")
            data.append(line.split(" "))
    return data


"""
Mark the path
"""


def markPixel():
    return [152, 9, 218]


class Orienteering:
    __slots__ = "elevations", "np_arr", "points", "penalties", "output_filename", "path"

    def __init__(self, terr_filename, elev_filename, path_file, output_filename):
        self.elevations = readElevations(elev_filename)
        self.np_arr = loadImage(terr_filename)
        self.points = readPoints(path_file)
        self.output_filename = output_filename
        self.path = set()
        # speed on different terrain type
        self.penalties = {
            "#F89412": 5,
            "#FFC000": 4,
            "#FFFFFF": 6,
            "#02D03C": 3.5,
            "#028828": 3.5,
            "#054918": 0.1,
            "#0000FF": 2,
            "#473303": 5,
            "#000000": 10,
            "#CD0065": 0.0001
        }
        self.process()

    """
    Calculate the optimal distance for all the points to be visited and map the path on output image
    """

    def process(self):
        distance = 0
        for i in range(len(self.points) - 1):
            distance += self.A_star(self.points[i], self.points[i + 1])
        for i, j in self.path:
            self.np_arr[i][j] = markPixel()

        for point in self.points:
            self.markPoint(point)

        pil_image = Image.fromarray(self.np_arr)
        pil_image.show()
        pil_image.save(self.output_filename)
        print("Total Distance: " + str(distance) + " m")

    """
    A* Algorithm for start to end node
    """

    def A_star(self, start, end):
        visited = set()
        visited.add(start)
        closed = set()

        g = {start: 0}

        f = {}

        parents = {start: start}
        heap = []
        heapq.heappush(heap, (g[start] + self.getHeuristic(start, end), start))

        while len(visited) > 0:
            n = None
            distance = 0

            f[n], n = heapq.heappop(heap)

            if n == end:
                while parents[n] != n:
                    i, j = n
                    # self.np_arr[i][j] = markPixel()
                    self.path.add(n)
                    n = parents[n]
                    distance += self.getDistance((i, j), n)
                self.path.add(n)
                return distance

            for m in self.get_neighbors(n):
                if m not in visited and m not in closed:
                    visited.add(m)
                    parents[m] = n
                    g[m] = g[n] + self.getTime(n, m)
                    f[m] = g[m] + self.getHeuristic(m, end)
                    heapq.heappush(heap, (f[m], m))
                else:
                    if g[m] > g[n] + self.getTime(n, m):
                        g[m] = g[n] + self.getTime(n, m)
                        f[m] = g[m] + self.getHeuristic(m, end)
                        parents[m] = n

                        if m in closed:
                            closed.remove(m)
                            visited.add(m)
                            heapq.heappush(heap, (f[m], m))

            visited.remove(n)
            closed.add(n)

        return None

    """
    Calculates the euclidean distance from start to end node and speed based on end node's terrain. And then calculates 
    the time to reach from start to end node. 
    """

    def getHeuristic(self, start, end):
        x_start, y_start = start
        x_end, y_end = end
        r, g, b = self.np_arr[x_end][y_end][:3]
        try:
            speed = self.penalties[rgb2hex(r, g, b)]
        except KeyError:
            speed = 0.1
        return math.sqrt(((x_end - x_start) ** 2) + ((y_end - y_start) ** 2)) / speed

    """
    Calculates time from start to end node considering the elevation difference.
    """

    def getTime(self, start, end):
        x_start, y_start = start
        x_end, y_end = end
        r, g, b = self.np_arr[x_end][y_end][:3]
        try:
            speed = self.penalties[rgb2hex(r, g, b)]
        except KeyError:
            speed = 0.1
        temp = (float(self.elevations[x_start][y_start]) - float(self.elevations[x_end][y_end])) ** 2
        return (math.sqrt(((x_end - x_start) ** 2) + ((y_end - y_start) ** 2) + temp)) / speed

    """
    Returns the distance from start to end node
    """

    def getDistance(self, start, end):
        x_start, y_start = start
        x_end, y_end = end
        x_start, y_start = x_start, y_start
        x_end, y_end = x_end, y_end
        temp = ((float(self.elevations[x_start][y_start]) - float(self.elevations[x_end][y_end])) ** 2)
        return math.sqrt((((x_end - x_start) * 7.55) ** 2) + (((y_end - y_start) * 10.29) ** 2) + temp)

    """
    returns the 8 neighboring points of a node.
    """

    def get_neighbors(self, node):
        x, y = node
        nbrs = []
        if self.check(x - 1, y - 1):
            nbrs.append((x - 1, y - 1))
        if self.check(x, y - 1):
            nbrs.append((x, y - 1))
        if self.check(x + 1, y - 1):
            nbrs.append((x + 1, y - 1))
        if self.check(x + 1, y):
            nbrs.append((x + 1, y))
        if self.check(x + 1, y + 1):
            nbrs.append((x + 1, y + 1))
        if self.check(x, y + 1):
            nbrs.append((x, y + 1))
        if self.check(x - 1, y + 1):
            nbrs.append((x - 1, y + 1))
        if self.check(x - 1, y):
            nbrs.append((x - 1, y))
        return nbrs

    """
    Mark the nodes
    """

    def markPoint(self, point):
        x, y = point
        if self.check(x,y): self.np_arr[x][y] = [227, 3, 10]
        if self.check(x-1,y-1): self.np_arr[x - 1][y - 1] = [227, 3, 10]
        if self.check(x,y-1): self.np_arr[x][y - 1] = [227, 3, 10]
        if self.check(x+1,y-1): self.np_arr[x + 1][y - 1] = [227, 3, 10]
        if self.check(x+1,y): self.np_arr[x + 1][y] = [227, 3, 10]
        if self.check(x+1,y+1): self.np_arr[x + 1][y + 1] = [227, 3, 10]
        if self.check(x,y+1): self.np_arr[x][y + 1] = [227, 3, 10]
        if self.check(x-1,y+1): self.np_arr[x - 1][y + 1] = [227, 3, 10]
        if self.check(x-1,y): self.np_arr[x - 1][y] = [227, 3, 10]


    """
    Check if the points lie in the image boundary.
    """
    def check(self, x, y):
        if 0 <= x < len(self.np_arr) and 0 <= y < len(self.np_arr[0]):
            return True
        return False


if __name__ == '__main__':
    terr_filename = sys.argv[1]
    elev_filename = sys.argv[2]
    path_file = sys.argv[3]
    output_filename = sys.argv[4]
    Orienteering(terr_filename, elev_filename, path_file, output_filename)
