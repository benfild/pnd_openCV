import cv2
import pytesseract
from scipy import ndimage

# read the image file and convert in grayscale Image
img = cv2.imread('carsImages/WhatsApp Image 2020-10-15 at 11.23.42.jpeg')
imgGs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgGS = cv2.GaussianBlur(imgGs, (5, 5), 0)

# Edges detection
imgEdged = cv2.Canny(imgGs, 63, 200)
cv2.imshow("Edged Image", imgEdged)
cv2.waitKey(0)


# find contours based on edges
contours, new = cv2.findContours(imgEdged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

# initialize licence plate contour and x, y coordinates
contour_with_LP = None
license_plate = None
x = None
y = None
w = None
h = None

# find the contour with 4 corners and create ROI around it
for contour in contours:
    # find parameter of contour and it should be a closed contour
    parameter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.1 * parameter, True)
    if len(approx) == 4:  # see whether it is rectangle
        contour_with_LP = approx
        x, y, w, h = cv2.boundingRect(contour)
        license_plate = ndimage.maximum_filter(imgGs[y:y + h, x:x + w], size=3)
        print("Plate Detected")
        cv2.imshow("Detected plate", license_plate)
        cv2.waitKey(0)
        break
    else:
        print("No Plate Detected")