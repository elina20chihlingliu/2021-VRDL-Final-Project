import cv2
img = cv2.imread("./ALB/img_00003.jpg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(img.shape)

h = 107
w =  353
x = 377
y =  66

"""
(x,y):top-left coordinate
(x+w,y+h):down-right coordinate
"""

cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),6)
cv2.imshow("name",img)
cv2.waitKey(0)