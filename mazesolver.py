import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from skimage.segmentation import clear_border
from customFunctions import erode

def resize_image_and_show(image):
    preRezised = image
    scale_percent = 20
    # width = int(preRezised.shape[1] * scale_percent / 100)
    # height = int(preRezised.shape[0] * scale_percent / 100)
    width = int(1000)
    height = int(1000)
    dim = (width, height)
    Resized = cv.resize(preRezised, dim, interpolation=cv.INTER_AREA)

    # Show scaled image
    # cv.imshow("Resized", Resized)
    return Resized
    # cv.waitKey(0)
    # cv.destroyAllWindows()

# Own implementation of matlab find function to find all white corner pixels in image
def find_coords_box(image):
    c1, c2, c3, c4 = 0, 0, 0, 0
    list_of_white_pixels = []

    # make multithreaded with image sections
    
    for y in range(0,image.shape[0],5):
        for x in range(0,image.shape[1],5):
            if image[y,x] == 255:
                list_of_white_pixels.append([x,y])
    
    # Find top left
    c1 = c2 = c3 = c4 = list_of_white_pixels[0]

    for i in range(len(list_of_white_pixels)):
        x = list_of_white_pixels[i][0]
        y = list_of_white_pixels[i][1]
        if (y + x) < (c1[1] + c1[0]):
            c1 = list_of_white_pixels[i]
        if (y - x) < (c2[1] - c2[0]):
            c2 = list_of_white_pixels[i]
        if (y + x) > (c3[1] + c3[0]):
            c3 = list_of_white_pixels[i]
        if (y - x) > (c4[1] - c4[0]):
            c4 = list_of_white_pixels[i]
    
    return c1, c2, c3, c4

def process_image(image):

    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # Apply threshold
    # ret, thresh = cv.threshold(grayscale, 45, 255, 0)
    ret, thresh = cv.threshold(grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    thresh = cv.bitwise_not(thresh)

    # open thresh
    kernel = np.ones((3,3), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=3)
    
    # dilate thresh
    eroded = erode(dilated, 3, iterations=3)

    
    eroded = clear_border(eroded)

    # create big kernel
    kernel = np.ones((11,11), np.uint8)
    
    # dilate eroded 10 times
    dilated = cv.dilate(eroded, kernel, iterations=6)
    

    nb_component, output, stats, centroids = cv.connectedComponentsWithStats(dilated, connectivity=8)

    sizes = stats[1:, -1]; nb_component = nb_component - 1

    min_size = 1000000

    img2 = np.zeros((output.shape))

    for i in range(0, nb_component):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    new = cv.bitwise_and(np.uint8(img2), eroded)

    originalPoints = find_coords_box(img2)

    Left = int((originalPoints[0][0] + originalPoints[3][0]) / 2 + 1000)
    Right = int((originalPoints[1][0] + originalPoints[2][0]) / 2 + 1000)
    Top = int((originalPoints[0][1] + originalPoints[1][1]) / 2 + 1000)
    Bottom = int((originalPoints[2][1] + originalPoints[3][1]) / 2 + 1000)

    pts1 = np.float32([originalPoints[0], originalPoints[1], originalPoints[2], originalPoints[3]])
    pts2 = np.float32([(Left, Top),(Right, Top),(Right, Bottom),(Left, Bottom)])

    M = cv.getPerspectiveTransform(pts1,pts2)

    warped = cv.warpPerspective(new,M,(Right, Bottom))
    warped2 = cv.warpPerspective(image,M,(image.shape[1] * 2, image.shape[0] * 2))
    ret, IM = cv.invert(M)
    
    # crop to only the box
    warped = warped[Top:Bottom, Left:Right]
    
    # cv.imshow("warped", resize_image_and_show(warped2))

    # make warped black and white
    ret, warped = cv.threshold(warped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    mazeHeight, mazeWidth = warped.shape
    approxWallWidth = mazeWidth // 21
    approxWallHeight = mazeHeight // 21
    print(warped.shape)

    warped = ~warped[0+approxWallHeight:mazeHeight-approxWallHeight, 0+approxWallWidth:mazeWidth-approxWallWidth]
    print(warped.shape)
    # look for white pixel in top row
    # if not found, rotate image 90 degrees
    rotated = 0
    for x in range(warped.shape[1]):
        if warped[0,x] == 255:
            break
    else:
        warped = np.rot90(warped)
        print("Rotated 90 degrees")
        rotated = 1

    

    solved = solve_maze(warped)

    if rotated:
        solved = np.rot90(solved, 3)

    
    combined = cv.bitwise_xor(warped2[Top+approxWallHeight:Bottom-approxWallHeight, Left+approxWallWidth:Right-approxWallWidth], solved)
    warped2[Top+approxWallHeight:Bottom-approxWallHeight, Left+approxWallWidth:Right-approxWallWidth] = combined
    warped3 = cv.warpPerspective(warped2,IM,(image.shape[1], image.shape[0]))

    return warped3

def find_start(maze):
    for x in range(maze.shape[1]):
        if maze[0,x] == 255:
            return (x,0)

def find_end(maze):
    for x in range(maze.shape[1]):
        if maze[maze.shape[0]-1,x] == 255:
            return (x,maze.shape[0]-1)

def find_path(maze, start, end):
    UP      = 1
    RIGHT   = 2
    DOWN    = 3
    LEFT    = 4
    WALL    = 0
    PATH    = 255

    PENCIL_X = start[0]
    PENCIL_Y = start[1]

    PENCIL_DIRECTION = DOWN

    # create a new canvas
    canvas = np.zeros((maze.shape[0], maze.shape[1], 3), np.uint8)
    print("Canvas Dimentions: ", canvas.shape)
    print("Maze Dimentions: ", maze.shape)



    while (PENCIL_X,PENCIL_Y) != end:

        PixelLeft = maze[PENCIL_Y, PENCIL_X - 1]
        PixelRight = maze[PENCIL_Y, PENCIL_X + 1]
        PixelUp = maze[PENCIL_Y - 1, PENCIL_X]
        PixelDown = maze[PENCIL_Y + 1, PENCIL_X]

        # Switch case for direction
        if PENCIL_DIRECTION == UP:
            if PixelUp == WALL and PixelLeft == WALL and PixelRight == WALL:
                PENCIL_DIRECTION = DOWN
            elif PixelUp == PATH and PixelRight == WALL:
                PENCIL_DIRECTION = UP
            elif PixelUp == WALL and PixelRight == WALL:
                PENCIL_DIRECTION = LEFT
            elif PixelUp == PATH and PixelRight == PATH:
                PENCIL_DIRECTION = RIGHT
            elif PixelUp == WALL and PixelRight == PATH:
                PENCIL_DIRECTION = RIGHT

        elif PENCIL_DIRECTION == RIGHT:
            if PixelRight == WALL and PixelUp == WALL and PixelDown == WALL:
                PENCIL_DIRECTION = LEFT
            elif PixelRight == PATH and PixelDown == WALL:
                PENCIL_DIRECTION = RIGHT
            elif PixelRight == WALL and PixelDown == WALL:
                PENCIL_DIRECTION = UP
            elif PixelRight == PATH and PixelDown == PATH:
                PENCIL_DIRECTION = DOWN
            elif PixelRight == WALL and PixelDown == PATH:
                PENCIL_DIRECTION = DOWN

        elif PENCIL_DIRECTION == DOWN:
            if PixelDown == PATH and PixelLeft == WALL:
                PENCIL_DIRECTION = DOWN
            elif PixelDown == WALL and PixelLeft == WALL:
                if PixelRight == WALL:
                    PENCIL_DIRECTION = UP
                else:
                    PENCIL_DIRECTION = RIGHT
            elif PixelDown == PATH and PixelLeft == PATH:
                PENCIL_DIRECTION = LEFT
            elif PixelDown == WALL and PixelLeft == PATH:
                PENCIL_DIRECTION = LEFT

        elif PENCIL_DIRECTION == LEFT:
            if PixelLeft == WALL and PixelUp == WALL and PixelDown == WALL:
                PENCIL_DIRECTION = RIGHT
            elif PixelLeft == PATH and PixelUp == WALL:
                PENCIL_DIRECTION = LEFT
            elif PixelLeft == WALL and PixelUp == WALL:
                PENCIL_DIRECTION = DOWN
            elif PixelLeft == PATH and PixelUp == PATH:
                PENCIL_DIRECTION = UP
            elif PixelLeft == WALL and PixelUp == PATH:
                PENCIL_DIRECTION = UP
        if(maze[PENCIL_Y+1, PENCIL_X+1] == PATH):
            canvas[PENCIL_Y:PENCIL_Y + 10, PENCIL_X:PENCIL_X + 10, [0,1]] = PATH
            canvas[PENCIL_Y:PENCIL_Y + 10, PENCIL_X:PENCIL_X + 10, [2]] = WALL
        else:
            canvas[PENCIL_Y:PENCIL_Y + 10, PENCIL_X:PENCIL_X + 10, [0,1]] = WALL
            canvas[PENCIL_Y:PENCIL_Y + 10, PENCIL_X:PENCIL_X + 10, [2]] = PATH

        # Move the pencil
        if PENCIL_DIRECTION == UP:
            PENCIL_Y -= 1
        elif PENCIL_DIRECTION == RIGHT:
            PENCIL_X += 1
        elif PENCIL_DIRECTION == DOWN:
            PENCIL_Y += 1
        elif PENCIL_DIRECTION == LEFT:
            PENCIL_X -= 1

    # add two dimentions to maze
    maze = np.expand_dims(maze, axis=2)

    return canvas

def solve_maze(maze):
    # find start and end points
    start = find_start(maze)
    print("Start: ", start)
    end = find_end(maze)
    print("End: ", end)

    # find path
    path = find_path(maze, start, end)

    # return path
    return path

def run_maze_solver(image):
    # Load image

    processed_image = process_image(image)

    return cv.cvtColor(processed_image, cv.COLOR_BGR2RGB) 


if __name__ == "__main__":
    image = cv.imread("maze2.jpg")
    run_maze_solver(image)
    cv.waitKey(0)
    cv.destroyAllWindows()