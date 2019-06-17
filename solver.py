tmap = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],[1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],[0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],[0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

import copy

class Cell(object):
    """Cell class representing a position in the maze

    IMPORTANT: x is the ROWS DOWN from the top (cartesian -y coord)
               y is the COLUMNS ACROSS from the left (cartesian x coord)
        this is done so that accessing cells of a grid is consistent
        with python indexing

    isWall is a bool representing whether the cell is a wall or not
    parent is an optional param indicating the previous step in
        the path to this cell
    g is cost of the path from start to this cell
    h is heuristic function estimating cost of cheapest path
        from this import cell to the goal, should never overestimate
        in order to ensure admissibility
    """

    def __init__(self, x, y, isWall=0, parent=None):
        self.x = x
        self.y = y

        self.isWall = False if isWall == 0 else True
        self.parent = parent

        self.f = 1e10
        self.g = 1e10

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class AStar(object):
    """Master class representing the grid of cells
    start is the top left cell
    goal is the bottom right cell
    openSet is a list of all positions to be considered
        only the starting cell upon instantiation
    closedSet is a list of positions that should NOT be considered further
    """

    def __init__(self, grid):
        self.grid = self._cellify(grid)
        self.start = self.grid[0][0]
        self.goal = self.grid[-1][-1]
        self.closedSet = []
        self.openSet = [self.start]

    def _cellify(self, grid):
        """Turn initial grid into a grid of cell objects
        called only during instantiation
        """
        newgrid = []
        for row_idx, row_val in enumerate(grid):
            newrow = []
            for cell_idx, cell_val in enumerate(row_val):
                newrow.append(Cell(row_idx, cell_idx, isWall=cell_val))
            newgrid.append(newrow)
        return newgrid

    def get_neighbours(self, cell):
        """Return all neighbours of a given cell (top, bottom, left, right)
        given the conditions that
        1) neighbour cell is not out of bounds
        2) neighbour cell is not a wall
        """
        neighbours = []

        for i in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
            tNeighbour = cell.x + i[0], cell.y + i[1]
            if 0 <= tNeighbour[0] < len(self.grid) and \
               0 <= tNeighbour[1] < len(self.grid[0]):
                neighbours.append(self.grid[tNeighbour[0]][tNeighbour[1]])

        return [i for i in neighbours if i.isWall == False]


def hScore(cell1, cell2):
    """Returns Euclidean distance between two cells
    which will always be less than the actual distance travelled
    serves as the heuristic function for the A* algorithm
    """
    return (abs(cell1.x - cell2.x)**2 + abs(cell1.y - cell2.y)**2)**0.5


def retracePath(cell):
    """Called when maze is solved
    becomes the parent of the cell until there are no more parents
    returns the number of steps from start to end
    """
    counter = 1
    curr = cell
    while curr.parent is not None:
        counter += 1
        curr = curr.parent
    return counter


def solve(maze):
    """Solve the maze based on the A* algorithm
    """
    start = maze.start
    goal = maze.goal

    # start with only the beginning cell under consideration
    openSet = [start]
    # no cells have been considered already, so empty
    closedSet = []

    # distance from start to start is 0
    start.g = 0
    # f score for first cell is entirely heuristic
    start.f = hScore(start, goal)

    # while there are cells to consider
    while len(openSet) > 0:
        # sort open set so that the lowest f score is at the end (faster)
        openSet.sort(key=lambda x: x.f, reverse=True)
        # get the node with lowest f score
        curr = openSet[-1]
        # if solved, return number of steps taken
        if curr == goal:
            return retracePath(curr)

        # remove last item in openSet (curr) and add it to closedSet
        closedSet.append(openSet.pop())

        # consider neighbours
        for neighbour in maze.get_neighbours(curr):

            # dont want to consider cells in closed set (again)
            if neighbour in closedSet:
                continue

            # distance between current and neighbour is 1
            tentative_gScore = curr.g + 1

            # add neighbour to openset for consideration next loop
            if neighbour not in openSet:
                openSet.append(neighbour)

            # do not consider suboptimal paths
            elif tentative_gScore >= neighbour.g:
                continue

            # record path taken, update scores
            neighbour.parent = curr
            neighbour.g = tentative_gScore
            neighbour.f = neighbour.g + hScore(neighbour, goal)


def mazeGen(maze):
    """Generates all possible variations of the maze with
    a single wall removed. Does not return original maze since
    a wall with a removed wall is always at least as good as
    a maze without removed walls.
    """

    # get the positions of the 1s in the maze
    walls = []
    for row_idx, row_val in enumerate(maze):
        for col_idx, col_val in enumerate(row_val):
            if maze[row_idx][col_idx] == 1:
                walls.append((row_idx, col_idx))

    # destroy a wall and return the variation
    for wall in walls:
        newmaze = copy.deepcopy(maze)
        newmaze[wall[0]][wall[1]] = 0
        yield newmaze


def solution(s):
    """Solve each variation of the maze and return the lowest score
    """

    lowest = 1e10
    allmazes = mazeGen(s)
    for m in allmazes:
        maze = AStar(m)
        steps = solve(maze)
        if steps is not None:
            lowest = min(lowest, steps)
    return lowest

print(solution(tmap))
