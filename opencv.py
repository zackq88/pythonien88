cv2.imread()=img 
cv2.imshow()
cv2.waitkey(0)
img.resize()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)-->to gray
cap=cv2.VideoCapture()
cv2.putText(..)

cap=cv2.VideoCapture("..")
While True:
  sucess,img=cap.read()
  cv2.imshow("video",img)
  if cv2.waitKey(1) & 0xFF==ord('q'):
    break
