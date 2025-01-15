import cv2

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Baca gambar
img = cv2.imread('assets/sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Deteksi wajah
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# Gambarkan persegi di sekitar wajah yang terdeteksi
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Tampilkan gambar dengan wajah yang dilingkari
cv2.imshow('Deteksi Wajah', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
