import cv2
import numpy as np
from tensorflow.keras.models import load_model

#smaller = less blue
BLUR_FACTOR = 3 
#0-255, smaller = more sensitive to light
THRESHOLD_VALUE = 115
#110-130 worked best with a grey
#minimum pixel size to be detected
MIN_CHARACTER_SIZE = 10
#value between 0 and 1, decides wether confident enough to output
CONFIDENCE_THRESHOLD = 0.8
#add extra space on sides to not cut off numbers
PADDING = 10
#
BOX_COLOR = (0,255,0) #green
TEXT_COLOR = (0,0,255) #red
#
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

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
    
    #Makes darks colours bright white and light colours black
    _, threshold = cv2.threshold(gray, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV)
    return threshold

def roi_reshaping(threshold, x1, y1, x2, y2):
    roi = threshold[y1:y2, x1:x2]
    roi = cv2.resize(roi, (28,28))
    roi = roi.astype(np.float32) / 255.0
    roi = roi.reshape(1, 28,28, 1)
    return roi

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
    
    #finds shapes in the image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #sorts shapes from left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])
    
    #goes through each shape, removes small shapes (noise) and draws rectangles around remaining shapes
    for shapes in contours:
        x, y, width, height = cv2.boundingRect(shapes)

        x1 = max(0, x - PADDING)
        y1 = max(0, y - PADDING)
        x2 = min(frame.shape[1], x + width + PADDING)
        y2 = min(frame.shape[0], y + height + PADDING)

        aspect_ratio = width / height

        if width > MIN_CHARACTER_SIZE and height > MIN_CHARACTER_SIZE and 0.2 < aspect_ratio < 2.0:
            roi = roi_reshaping(threshold, x1, y1, x2, y2)
            number_recognition(model, roi, expression)

    cv2.imshow("math detector", frame)
    cv2.imshow("threshold", threshold)
    
    #exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()