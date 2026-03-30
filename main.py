import cv2
import numpy as np
from tensorflow.keras.models import load_model

#1. intersection error, when 8 is written the cross over between the lines creates a small white dot and it detects 8 as two seperate numbers.
#2. Confidence is too high for number 6, when 6 written is usually gets it right but is only usually ~60% cinfident, so lower confidence required.

# smaller = less blue
BLUR_FACTOR = 3 
# BLOCKSIZE = area to claculate light intesnity in
# C = sensitivity (higher = more sensitive, fewer pixels pass)
BLOCKSIZE = 11
C=15
# minimum pixel size to be detected
MIN_CHARACTER_SIZE = 10
# value between 0 and 1, decides wether confident enough to output
CONFIDENCE_THRESHOLD = 0.7
# add extra space on sides to not cut off numbers, scales with the font size
PADDING_MULTIPLIER = 0.1
#
BOX_COLOR = (0,255,0) #green
TEXT_COLOR = (0,0,255) #red
#possible values a shape can be classified as
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
#Pixels close enough to make same colour to fill in lighter gaps of pen strokes
MERGE_RANGE = 6

expression = ""

def start_model(load_model):
    model = load_model('digit_model.h5')
    print("Model loaded, starting webcam...")
    return model

def image_processing(frame):
    #blurs the image to reduce noise
    blurred = cv2.GaussianBlur(frame, (BLUR_FACTOR, BLUR_FACTOR), 0)
    #sets to greyscale
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    #Makes images inverted black and white, adaptive so adjusts to light intensity in areas of the screen
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, BLOCKSIZE, C)
    return threshold

def close_small_gaps(threshold, MERGE_RANGE):
    #creates a square by the size of MERGE_RANGE
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MERGE_RANGE, MERGE_RANGE))
    #puts the new box in the gap
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
    return threshold

def roi_reshaping(threshold, x1, y1, x2, y2):
    roi = threshold[y1:y2, x1:x2]
    roi = cv2.resize(roi, (28,28))
    roi = roi.astype(np.float32) / 255.0
    roi = roi.reshape(1, 28,28, 1)
    return roi

def is_inside(box1, box2):
    x1_a, y1_a, x2_a, y2_a = box1
    x1_b, y1_b, x2_b, y2_b = box2

    centre_x = (x1_a + x2_a) / 2
    centre_y = (y1_a + y2_a) / 2
    return x1_b <= centre_x <= x2_b and y1_b <= centre_y <= y2_b

#removes boxes that are inside other boxes, used to fix intersection error
def filter_boxes(boxes):
    filtered_boxes = []
    for i, box in enumerate(boxes):
        contained = False
        for j, other_box in enumerate(boxes):
            if i !=j and is_inside(box, other_box):
                contained = True
                break
        if not contained and is_inside(box, other_box):
            filtered_boxes.append(box)
    return filtered_boxes    

def number_recognition(model, roi, expression):
    prediction = model.predict(roi, verbose=0)
    label = LABELS[np.argmax(prediction)]
    confidence = np.max(prediction)

    if confidence > CONFIDENCE_THRESHOLD:
        expression += label
    else:
        label = "?"
    
    cv2.rectangle(frame, (x1,y1), (x2, y2), BOX_COLOR, 2)
    cv2.putText(frame,f"{label} {confidence:.0%}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    
model = start_model(load_model)  

webcam = cv2.VideoCapture(0) #takes input from webcam

while True:

    _, frame = webcam.read(0) #ret - boolean value indicating if the frame was read successfully
    
    threshold = image_processing(frame)

    #Closes small gaps in the contours to make it more likely to detect numbers as one shape instead of multiple shapes
    threshold = close_small_gaps(threshold, MERGE_RANGE)
    #finds shapes in the image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sorts shapes from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    # list of all boxes in the image, used to check if a box is inside another box
    boxes = []
    
    #goes through each shape, removes small shapes (noise) and draws rectangles around remaining shapes
    for shapes in contours:
        x, y, width, height = cv2.boundingRect(shapes)

        aspect_ratio = width / height

        #if the shapes are big enough to be a number
        if width > MIN_CHARACTER_SIZE and height > MIN_CHARACTER_SIZE and 0.2 < aspect_ratio < 2.0:
            
            x_padding = int(width * PADDING_MULTIPLIER)
            y_padding = int(height * PADDING_MULTIPLIER)

            #adds padding to not cut off nmbers
            x1 = max(0, x - x_padding) #left edge
            y1 = max(0, y - y_padding) #top edge
            x2 = min(frame.shape[1], x + width + x_padding) #right edge
            y2 = min(frame.shape[0], y + height + y_padding) #bottom edge

            #adds box to list of boxes
            boxes.append((x1, y1, x2, y2))

            list_of_filtered_boxes = filter_boxes(boxes)

            for box in list_of_filtered_boxes:
                #draws rectangles around detected shapes
                roi = roi_reshaping(threshold, x1, y1, x2, y2)
                number_recognition(model, roi, expression)

    cv2.imshow("math detector", frame)
    cv2.imshow("threshold", threshold)
    
    #exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()