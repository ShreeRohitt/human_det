import cv2
import tkinter as tk
from tkinter import messagebox
import imutils

#we are creating a function to put up a pop message using tinker
def show_popup():
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo("Human Detected", "A human has been detected!")
    root.destroy()

# To open the live camera and 0 for defualt camera
cap = cv2.VideoCapture(0)  

# we are loading the HOG Descriptor which is a feature descriptor which is used for object detection to get the gradients.
hog = cv2.HOGDescriptor()

# we are integrating HOG Descriptor with SVM. So, svm is a machile learning algorithm and here we are using it for classification.
# In our case, we are using to detect whether human is present or not

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# we need to define the bounding box 

#x and y co-ordinates of top left corner
box_top_left = (100, 100)  
# x and y co-ordinates of bottom right corner
box_bottom_right = (500, 500) 

# first we are ssetting the flag human_detected as False
human_detected = False

# in order to run the camera detection process throughout, we are setting infinite loop here

while True:
    # to read the frame that we capture
    ret, frame = cap.read()
    #if we couldn't read

    if not ret:
        break
    
    # we are using imutils to ressizze the frame we read to process the compution fasterand this can be avoided too.
    frame = imutils.resize(frame, width=min(800, frame.shape[1]))
    
    # to detect human. This is the hog function to return a tuple when it detects the human
    (humans, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
    

    human_detected = False
    # we are checking whether the tuple we got belongs in the bounding box
    for (x, y, w, h) in humans:
        # index 0 and 1 are the co-ordinates of the corner
        if (box_top_left[0] < x < box_bottom_right[0] and #to check left side is within the horizontal 
            box_top_left[0] < x + w < box_bottom_right[0] and # to check the right side (+ width) is within the horizontal
            box_top_left[1] < y < box_bottom_right[1] and  #to check the upper is within the y axis
            box_top_left[1] < y + h < box_bottom_right[1]): # to check the lower is withing the y axis
            human_detected = True
        
        # to draw bounding box for human
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # the pop_up function when human detected is true.
    if human_detected:
        show_popup()
    
    # to make the bounding box fixed
    cv2.rectangle(frame, box_top_left, box_bottom_right, (255, 0, 0), 2)
    
    cv2.imshow('Human Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()