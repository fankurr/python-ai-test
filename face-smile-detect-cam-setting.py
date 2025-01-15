import cv2
import tkinter 
from tkinter import ttk

# Load Haar Cascade untuk deteksi wajah dan senyuman
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

# Buka kamera (0 adalah ID kamera default)
cap = cv2.VideoCapture(0)

# Variabel untuk menyimpan poin dan status senyuman
smile_points = 0
smile_detected = False  # Status apakah senyuman sedang terdeteksi

# Parameter default
resize_width = 300
resize_height = 300
scale_factor = 2.5
min_neighbors = 30
min_size = (40, 40)

# Fungsi untuk memperbarui parameter dari UI
def update_parameters():
    global resize_width, resize_height, scale_factor, min_neighbors, min_size
    try:
        resize_width = int(entry_resize_width.get())
        resize_height = int(entry_resize_height.get())
        scale_factor = float(entry_scale_factor.get())
        min_neighbors = int(entry_min_neighbors.get())
        min_size_width = int(entry_min_size_width.get())
        min_size_height = int(entry_min_size_height.get())
        min_size = (min_size_width, min_size_height)
        label_status.config(text="Parameters updated successfully!", fg="green")
    except ValueError:
        label_status.config(text="Invalid input! Check your values.", fg="red")

# Fungsi untuk membuka UI pengaturan
def open_settings_ui():
    ui = tkinter.Toplevel()
    ui.title("Smile Detection Parameters")

    # Resize Width
    tkinter.Label(ui, text="Resize Width:").grid(row=0, column=0, padx=5, pady=5)
    entry_resize_width = tkinter.Entry(ui)
    entry_resize_width.grid(row=0, column=1, padx=5, pady=5)
    entry_resize_width.insert(0, str(resize_width))

    # Resize Height
    tkinter.Label(ui, text="Resize Height:").grid(row=1, column=0, padx=5, pady=5)
    entry_resize_height = tkinter.Entry(ui)
    entry_resize_height.grid(row=1, column=1, padx=5, pady=5)
    entry_resize_height.insert(0, str(resize_height))

    # Scale Factor
    tkinter.Label(ui, text="Scale Factor:").grid(row=2, column=0, padx=5, pady=5)
    entry_scale_factor = tkinter.Entry(ui)
    entry_scale_factor.grid(row=2, column=1, padx=5, pady=5)
    entry_scale_factor.insert(0, str(scale_factor))

    # Min Neighbors
    tkinter.Label(ui, text="Min Neighbors:").grid(row=3, column=0, padx=5, pady=5)
    entry_min_neighbors = tk.Entry(ui)
    entry_min_neighbors.grid(row=3, column=1, padx=5, pady=5)
    entry_min_neighbors.insert(0, str(min_neighbors))

    # Min Size Width
    tkinter.Label(ui, text="Min Size Width:").grid(row=4, column=0, padx=5, pady=5)
    entry_min_size_width = tkinter.Entry(ui)
    entry_min_size_width.grid(row=4, column=1, padx=5, pady=5)
    entry_min_size_width.insert(0, str(min_size[0]))

    # Min Size Height
    tkinter.Label(ui, text="Min Size Height:").grid(row=5, column=0, padx=5, pady=5)
    entry_min_size_height = tkinter.Entry(ui)
    entry_min_size_height.grid(row=5, column=1, padx=5, pady=5)
    entry_min_size_height.insert(0, str(min_size[1]))

    # Tombol Update
    btn_update = tkinter.Button(ui, text="Update Parameters", command=update_parameters)
    btn_update.grid(row=6, column=0, columnspan=2, pady=10)

    # Status Label
    global label_status
    label_status = tkinter.Label(ui, text="", fg="green")
    label_status.grid(row=7, column=0, columnspan=2, pady=5)

# Buat jendela utama Tkinter
root = tkinter.Tk()
root.title("Smile Detection Camera")

# Tambahkan tombol untuk membuka UI pengaturan
btn_settings = tkinter.Button(root, text="Open Settings", command=open_settings_ui)
btn_settings.pack(pady=10)

# Jalankan UI dalam thread terpisah agar kamera tetap berjalan
import threading
threading.Thread(target=root.mainloop, daemon=True).start()

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
        face_roi_gray = cv2.resize(face_roi_gray, (resize_width, resize_height))
        
        # Deteksi senyuman dalam area wajah
        smiles = smile_cascade.detectMultiScale(
            face_roi_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size
        )

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
