import face_recognition
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Yüzleri Yükle --------------------
known_encodings = []
known_names = []

folder = "fotograf"
for filename in os.listdir(folder):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(folder, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(os.path.splitext(filename)[0])
            print(f"✅ Yüklendi: {filename}")
        else:
            print(f"⚠️ Yüz bulunamadı: {filename}")

# -------------------- Kamera Aç --------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("🚫 Kamera açılamadı.")
    exit()

print("🎥 Kamera açık. 'q' tuşuna pencere üzerindeyken basarak çıkabilirsin.")

# -------------------- Matplotlib Ayarları --------------------
plt.ion()  # interaktif mod aç
fig, ax = plt.subplots()
image_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # başlangıç boş ekran
plt.axis('off')
plt.title("Yüz Tanıma Sistemi")

# -------------------- Döngü --------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Görüntü alınamadı.")
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Bilinmiyor"
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = known_names[best_match]

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        image_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)

        # Çıkış kontrolü (matplotlib penceresi aktifken klavye dinlenebilir)
        if not plt.fignum_exists(fig.number):  # pencere kapatıldıysa
            print("🛑 Pencere kapatıldı. Çıkılıyor.")
            break

except Exception as e:
    print(f"⚠️ Hata oluştu: {e}")

# -------------------- Temizlik --------------------
cap.release()
plt.close('all')
