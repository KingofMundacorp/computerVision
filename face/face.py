import cv2
import os
imagePath = 'download2.jpg'

img = cv2.imread(imagePath)

print(img.shape)

output_dir = r'..\Wallpapers\cv2images'
save_path = os.path.join(output_dir, 'cv2image1.png')


gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(gray_image.shape)

if not os.path.exists(output_dir):
        os.makedirs(output_dir)

success = cv2.imwrite(save_path, gray_image)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

cv2.imwrite('img_rgb3.png',img_rgb)
