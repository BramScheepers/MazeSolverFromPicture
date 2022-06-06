# import the necessary packages
import tkinter
import tkinter.filedialog
from tkinter.ttk import LabelFrame
from PIL import Image
from PIL import ImageTk
from pathlib import Path
import cv2
from mazesolver import run_maze_solver

def select_image():
    # grab a reference to the image panels
    global panelA, panelB
    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkinter.filedialog.askopenfilename(
    parent=root, initialdir=Path.home() / "Pictures",
    title='Choose file',
    filetypes=[('jpg images', '.jpg'),
    ('png images', '.png')
    ]
    )
    # ensure a file path was selected
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        solved = run_maze_solver(image)
        baseWidth = 500
        wpercent = (baseWidth/image.shape[1])
        hsize = int((image.shape[0]*float(wpercent)))
        dim = (baseWidth,hsize)
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        solved = cv2.resize(solved, dim, interpolation = cv2.INTER_AREA)
        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # convert the images to PIL format...
        image = Image.fromarray(image)
        solved = Image.fromarray(solved)
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        solved = ImageTk.PhotoImage(solved)
        # if the panels are None, initialize them
        if panelA is None or panelB is None:
            panelA = tkinter.Label(leftlabelFrame, image=image)
            panelA.image = image
            panelA.grid()

            panelB = tkinter.Label(rightlabelFrame, image=solved)
            panelB.image = solved
            panelB.grid()
        else:
            panelA.configure(image=image)
            panelB.configure(image=solved)
            panelA.image = image
            panelB.image = solved
            # initialize the window toolkit along with the two image panels
root = tkinter.Tk()
root.title("Maze Solver")
# custom program icon
root.iconbitmap("maze.ico")

panelA = None
panelB = None

# create a left panel for the original image
leftlabelFrame = LabelFrame(root, text="Originele Doolhof", width=500, height=300)
leftlabelFrame.grid(row=0, column=0, padx=10, pady=2)
# create a right panel for the edge image
rightlabelFrame = LabelFrame(root, text="Opgeloste Doolhof", width=500, height=300)
rightlabelFrame.grid(row=0, column=1, padx=10, pady=2)
# create a button, then when pressed, will choose a folder
# and load the image
btn = tkinter.Button(root, text="Selecteer Doolhof", command=select_image)
btn.grid(row=1, column=0, columnspan=2, padx=10, pady=2)
# kick off the window's event loop
root.mainloop()

