import cv2

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Buka kamera (0 adalah ID kamera default)
cap = cv2.VideoCapture(0)

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Terapkan efek mirror
    frame = cv2.flip(frame, 1)  # Flip frame horizontal

    # Ubah frame ke grayscale untuk deteksi wajah
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambarkan persegi di sekitar wajah yang terdeteksi
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Tampilkan frame dengan kotak wajah dan efek mirror
    cv2.imshow('Deteksi Wajah dengan Efek Mirror', frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
