import cv2
import numpy as np
from matplotlib import pyplot as plt

#1) read image
image = cv2.imread("/Users/joudabuzaid/Desktop/testImage.JPEG");
cv2.imshow("Original", image)
cv2.waitKey(0)

#2) resize 16x16
res = cv2.resize(image,(16, 16), interpolation = cv2.INTER_CUBIC)
cv2.imshow("Resizing 16X16", res)
cv2.waitKey(0)

#3) convert to gray
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)
cv2.waitKey(0)

#4)convert to 0 if <100
Img=np.array(gray)
for i in range(16):
    for j in range(16):
        if Img[i][j]<100:
            Img[i][j]=0
cv2.imshow("less than 0", Img)
cv2.waitKey(0)

#5) matrix of pixel values
print("Img Pixel Value Matrix: ")
for i in range(16):
    for j in range(16):
        print(Img[i][j], end=" ")
    print()

#6) derivative x and y
Fy = np.array([[1.,2,1],[0,0,0],[-1,-2,-1]])
Fx = np.array([[-1,0,1.],[-2,0,2],[-1,0,1]])
Ix = cv2.filter2D(Img,cv2.CV_64F, Fx)
Iy = cv2.filter2D(Img,cv2.CV_64F, Fy)
mag1, ang = cv2.cartToPolar(Ix,Iy)
cv2.imshow("Magnitude",mag1);
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
mag,angle=cv2.cartToPolar(Sx,Sy,angleInDegrees=True)
cv2.imshow("Magnitude",mag);
cv2.imshow("Direction",angle);
cv2.waitKey();


#9)gradient magnitude and direction in the matrix form
print("Gradient Magnitude Matrix: ")
gmag=np.array(mag)
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
weak_th = None
strong_th = None
mag_max = np.max(mag)
print('gradient: max=',np.max(mag), ', min=',np.min(mag))
print('angle: max=',np.max(angle), ', min=',np.min(angle))
if not weak_th: weak_th = mag_max * 0.1
if not strong_th: strong_th = mag_max * 0.5

height, width = Img.shape
for i_x in range(width):
    for i_y in range(height):
        grad_ang = ang[i_y, i_x]
        grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

        # In the x axis direction
        # gradient angle <= 45/2 means angle is approximated as 0,
        if grad_ang <= 22.5:
            neighb_1_x, neighb_1_y = i_x - 1, i_y
            neighb_2_x, neighb_2_y = i_x + 1, i_y

        # top right (diagnol-1) direction,
        # angle is approx 45, so edge is left diagonal like "\", so look top-left and bot-right pixels
        elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
            neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
            neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

        # In y-axis direction
        # angle is approx 90, so edge is horizontal, so look right, left
        elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
            neighb_1_x, neighb_1_y = i_x, i_y - 1
            neighb_2_x, neighb_2_y = i_x, i_y + 1

        # top left (diagnol-2) direction
        # angle is approx 135, so edge is right diagonal like "/", so look top-right and bot-left
        elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
            neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
            neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

        # Now it restarts the cycle
        # angle is approx 180, so edge is vertical, so look up and down
        elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
            neighb_1_x, neighb_1_y = i_x - 1, i_y
            neighb_2_x, neighb_2_y = i_x + 1, i_y

            # Non-maximum suppression step
        if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
            if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                mag[i_y, i_x] = 0  # suppress the pixel as it is non-maximum as compared to its neighbors
                continue

        if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
            if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                mag[i_y, i_x] = 0
cv2.imshow('After suppression', mag)
cv2.waitKey()

# 11) non-max supp matrix
print("Non-maximal Suppression Matrix: ")
nonmax = np.array(mag)
for i in range(16):
    for j in range(16):
        print(nonmax[i][j], end=" ")
    print()

#12) hysterisis threshold
# double thresholding step
for i_x in range(width):
    for i_y in range(height):

        grad_mag = mag[i_y, i_x]

        if grad_mag < weak_th:
            mag[i_y, i_x] = 0
        elif strong_th > grad_mag >= weak_th:
            mag[i_y, i_x] = 2  # weak
        else:
            mag[i_y, i_x] = 1  # strong
M, N = Img.shape
for i in range(1, M - 1):
    for j in range(1, N - 1):
        if (mag[i, j] == 2):
            if ((mag[i + 1, j - 1] == 1) or (mag[i + 1, j] == 1) or (mag[i + 1, j + 1] == 1)
                    or (mag[i, j - 1] == 1) or (mag[i, j + 1] == 1)
                    or (mag[i - 1, j - 1] == 1) or (mag[i - 1, j] == 1) or (mag[i - 1, j + 1] == 1)):
                mag[i, j] = 1
            else:
                Img[i, j] = 0

cv2.imshow('After hysteresis', mag)
cv2.waitKey()

# 13) Hysterisis matrix
print("Hysterisis Matrix: ")
hyst = np.array(mag)
for i in range(16):
    for j in range(16):
        print(hyst[i][j], end=" ")
    print()

