import cv2
import numpy as np
from matplotlib import pyplot as plt

#1) read image
img = cv2.imread("/Users/joudabuzaid/Desktop/testImage.JPG");
cv2.imshow("Original", img)
cv2.waitKey(0)

#2) resize 16x16
res = cv2.resize(img,(16, 16), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Resizing 16x16", res)
cv2.waitKey(0)

#3) convert to gray
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

#4)convert to 0 if <100
arrimg=np.array(gray)
for i in range(16):
    for j in range(16):
        if arrimg[i][j]<100:
            arrimg[i][j]=0
cv2.imshow("less than 0", arrimg)
cv2.waitKey(0)

#5) matrix of pixel values
print("Pixel Matrix: ")
for i in range(16):
    for j in range(16):
        print(arrimg[i][j], end=" ")
    print()

#6) derivative x and y
Fy = np.array([[1.,2,1],[0,0,0],[-1,-2,-1]])
Fx = np.array([[-1,0,1.],[-2,0,2],[-1,0,1]])

Ix = cv2.filter2D(gray,cv2.CV_64F, Fx)
Iy = cv2.filter2D(gray,cv2.CV_64F, Fy)
mag, ang = cv2.cartToPolar(Ix,Iy)
cv2.imshow("Magnitude",mag);
cv2.imshow("Direction",ang);
cv2.waitKey();

#7) XY pixel value matrix
print("Derivative X Matrix: ")
x=np.array(Ix)
for i in range(16):
    for j in range(16):
        print(x[i][j], end=" ")
    print()

print("Derivative Y Matrix: ")
y=np.array(Iy)
for i in range(16):
    for j in range(16):
        print(y[i][j], end=" ")
    print()

#8) gradient magnitude
Sx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
Sy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
plt.subplot(1,2,1),plt.imshow(Sx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(Sy,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
mag2,angle=cv2.cartToPolar(Sx,Sy,angleInDegrees=True)
cv2.imshow("Magnitude",mag2);
cv2.imshow("Direction",angle);
cv2.waitKey();

#9)gradient magnitude and direction in the matrix form
print("Gradient Magnitude Matrix: ")
gmag=np.array(mag2)
for i in range(16):
    for j in range(16):
        print(gmag[i][j], end=" ")
    print()

print("Gradient Direction Matrix: ")
gang=np.array(angle)
for i in range(16):
    for j in range(16):
        print(gang[i][j], end=" ")
    print()


#10) non-maximal suppression of gradient magnitude
M, N = mag.shape
Non_max = np.zeros((M, N), dtype=np.uint8)

for i in range(1, M - 1):
    for j in range(1, N - 1):
        # Horizontal 0
        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180) or (-22.5 <= angle[i, j] < 0) or (-180 <= angle[i, j] < -157.5):
            b = mag[i, j + 1]
            c = mag[i, j - 1]
        # Diagonal 45
        elif (22.5 <= angle[i, j] < 67.5) or (-157.5 <= angle[i, j] < -112.5):
            b = mag[i + 1, j + 1]
            c = mag[i - 1, j - 1]
        # Vertical 90
        elif (67.5 <= angle[i, j] < 112.5) or (-112.5 <= angle[i, j] < -67.5):
            b = mag[i + 1, j]
            c = mag[i - 1, j]
        # Diagonal 135
        elif (112.5 <= angle[i, j] < 157.5) or (-67.5 <= angle[i, j] < -22.5):
            b = mag[i + 1, j - 1]
            c = mag[i - 1, j + 1]

            # Non-max Suppression
        if (mag[i, j] >= b) and (mag[i, j] >= c):
            Non_max[i, j] = mag[i, j]
        else:
            Non_max[i, j] = 0

cv2.imshow("Non-maximal Suppression",Non_max);
cv2.waitKey();

#11) non-max supp matrix
print("Non-maximal Suppression Matrix: ")
nonmax=np.array(Non_max)
for i in range(16):
    for j in range(16):
        print(Non_max[i][j], end=" ")
    print()

#12) hysterisis threshold
# Set high and low threshold
highThreshold = 21
lowThreshold = 15

m, n = Non_max.shape
out = np.zeros((m, n), dtype=np.uint8)

# If edge intensity is greater than 'High' it is a sure-edge
# below 'low' threshold, it is a sure non-edge
strong_i, strong_j = np.where(Non_max >= highThreshold)
zeros_i, zeros_j = np.where(Non_max < lowThreshold)

# weak edges
weak_i, weak_j = np.where((Non_max <= highThreshold) & (Non_max >= lowThreshold))

# Set same intensity value for all edge pixels
out[strong_i, strong_j] = 255
out[zeros_i, zeros_j] = 0
out[weak_i, weak_j] = 75
cv2.imshow("Hysterisis Threshold",out);
cv2.waitKey();

#13) Hysterisis matrix
print("Hysterisis Matrix: ")
hyst=np.array(out)
for i in range(16):
    for j in range(16):
        print(hyst[i][j], end=" ")
    print()
