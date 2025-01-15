import cv2

# Load Haar Cascade untuk deteksi wajah dan senyuman
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

# Buka kamera (0 adalah ID kamera default)
cap = cv2.VideoCapture(0)

# Variabel untuk menyimpan poin dan status senyuman
smile_points = 0
smile_detected = False  # Status apakah senyuman sedang terdeteksi

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()

    # Terapkan efek mirror
    frame = cv2.flip(frame, 1)  # Flip frame horizontal

    # Ubah frame ke grayscale untuk deteksi
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Gambarkan persegi di sekitar wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Area wajah untuk deteksi senyuman
        face_roi_gray = gray[y:y + h, x:x + w]
        face_roi_color = frame[y:y + h, x:x + w]

        # Perbesar area wajah
        # face_roi_gray = cv2.resize(face_roi_gray, (300, 300))
        # face_roi_color = cv2.resize(face_roi_gray, (300, 300))
        # Deteksi senyuman dalam area wajah
        smiles = smile_cascade.detectMultiScale(face_roi_gray, scaleFactor=2.5, minNeighbors=25, minSize=(25, 25))

        # Jika senyuman terdeteksi
        if len(smiles) > 0:
            if not smile_detected:  # Jika senyuman belum dihitung
                smile_points += 1
                smile_detected = True  # Tandai bahwa senyuman sedang dihitung
        else:
            smile_detected = False  # Reset status jika tidak ada senyuman

        # Gambarkan persegi di sekitar senyuman yang terdeteksi
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

    # Tampilkan poin di pojok kiri atas frame
    cv2.putText(
        frame,
        f'Smile Points: {smile_points}',
        (10, 30),  # Koordinat teks (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,  # Font
        1,  # Ukuran font
        (0, 255, 255),  # Warna teks (kuning)
        2,  # Ketebalan garis teks
        cv2.LINE_AA  # Gaya garis
    )

    # Tampilkan frame dengan kotak wajah dan senyuman
    cv2.imshow('Smile Points', frame)

    # Jika tombol 'q' ditekan, keluar dari loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
