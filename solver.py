import matplotlib.pyplot as plt

#testmap1 = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
#testmap1 = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
testmap1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],[1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],[1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],[0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1],[1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],[1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1],[0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],[0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],[0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0],[0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],[0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],[0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1],[1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]]

def nprint(lofl):
    for r in lofl:
        print((' ').join(map(str, r)))

nprint(testmap1)


# solve maze without removing any walls using a* search
class Cell(object):
    """ cell class representing a number/position in the maze

    created by passing in a position x, y, and an optional parent cell
    IMPORTANT: x is the ROWS DOWN from the top (cartesian -y coord)
               y is the COLUMNS ACROSS from the left (cartesian x coord)
        this is done so that accessing cells of a grid is consistent
        with python indexing
    g is cost of path from start to this cell
    h is heuristic function estimating cost of cheapest path
        from this import cell to the goal, should never overestimate
    cameFrom is reference to best path (previous node probably)
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
    def __init__(self, grid):
        self.grid = self.cellify(grid)
        self.start = self.grid[0][0]
        self.goal = self.grid[-1][-1]
        self.closedSet = []
        self.openSet = [self.start]

    def cellify(self, grid):
        """ Turn initial grid into a grid of cell objects
        """
        ngrid = []
        for row_idx, row_val in enumerate(grid):
            nrow = []
            for cell_idx, cell_val in enumerate(row_val):
                nrow.append(Cell(row_idx, cell_idx, isWall=cell_val))
            ngrid.append(nrow)
        return ngrid

    def get_neighbours(self, cell):
        """Return all four neighbours of a given cell
        if current cell is at left edge, don't return left neighbour
        because none exist, and etc. for all other edges
        do not return walls
        """
        curx, cury = cell.x, cell.y
        neighbours = []

        # current is not at top
        if curx != 0:
            neighbours.append(self.grid[curx-1][cury])
        # current is not at bottom
        if curx != len(self.grid) - 1:
            neighbours.append(self.grid[curx+1][cury])
        # current is not on left edge
        if cury != 0:
            neighbours.append(self.grid[curx][cury-1])
        # current is not on right edge
        if cury != len(self.grid[0]) - 1:
            neighbours.append(self.grid[curx][cury+1])

        return [i for i in neighbours if i.isWall == False]


def hScore(cell1, cell2):
    # returns manhattan distance with assumption of no walls
    return abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)

def retracePath(cell):
    # keeps returning parent cell until there are no more parents (start cell)
    counter = 1
    curr = cell
    print(curr.x, curr.y)
    plt.scatter(curr.y, -curr.x)
    while curr.parent is not None:
        curr = curr.parent
        print(curr.x, curr.y)
        plt.scatter(curr.y, -curr.x)
        counter += 1
    return 'solved in %s steps' %counter

def solve(maze):
    start = maze.start
    goal = maze.goal

    closedSet = []
    openSet = [start]
    start.g = 0
    start.f = hScore(start, goal)

    while len(openSet) > 0:
        # sort open set so that the lowest fscore is at the end
        openSet.sort(key=lambda x: x.f, reverse=True)
        # get the node with lowest f score
        curr = openSet[-1]
        #print(curr.x, curr.y)
        # if solved, print number of steps taken
        if curr == goal:
            #return 'solved'
            return retracePath(curr)

        # remove last item in openSet (curr) and add it to closedSet
        closedSet.append(openSet.pop())

        for neighbour in maze.get_neighbours(curr):

            #don't need this since walls arent returned as neighbours
            #if neighbour.isWall:
            #    closedSet.append(neighbour)

            if neighbour not in closedSet:

                # add one because the distance between current and
                # the neighbour is 1 step
                tentative_gScore = curr.g + 1

                # add neighbour to openset for consideration next loop
                if neighbour not in openSet:
                    openSet.append(neighbour)
                # if already a considered node but potential 
                # gscore is too high then dont do anything (suboptimal path)
                elif tentative_gScore >= neighbour.g:
                    pass

                neighbour.parent = curr
                neighbour.g = tentative_gScore
                neighbour.f = neighbour.g + hScore(neighbour, goal)


maze = AStar(testmap1)
print(solve(maze))
plt.show()







