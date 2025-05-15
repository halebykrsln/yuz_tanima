import face_recognition
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv

def log_giris(kisi_adi):                                                              
    with open("giris_kaydi.csv", mode="a", newline="", encoding="utf-8") as file:   
        writer = csv.writer(file)                                    
        now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        writer.writerow([now, kisi_adi])

def kaydet_bilinmeyen(frame, bbox, kisi_adi):
    if kisi_adi not in kaydedilen_bilinmeyen:
        if not os.path.exists("bilinmeyen"):
            os.makedirs("bilinmeyen")
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bilinmeyen/{now}.jpg"
        (left, top, right, bottom) = bbox
        cropped_face = frame[top:bottom, left:right]
        cv2.imwrite(filename, cropped_face)
        print(f"Bilinmeyen kişi kaydedildi: {filename}")
        kaydedilen_bilinmeyen.add(kisi_adi)

# -------------------- Yüzleri Yükle --------------------
known_encodings = []
kaydedilen_isimler = []
kaydedilen_kisiler = set()
kaydedilen_bilinmeyen = set()

folder = "fotograf"
for filename in os.listdir(folder):
    if filename.endswith((".jpg", ".png")):
        path = os.path.join(folder, filename)  
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)  
        if encoding:
            known_encodings.append(encoding[0])
            kaydedilen_isimler.append(os.path.splitext(filename)[0])
            print(f"Yüklendi: {filename}")
        else:
            print(f"Yüz bulunamadı: {filename}")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

print("Kamera açıldı. 'q' tuşuna basarak çıkış yapabilirsin.")

# -------------------- Matplotlib -------------------------
plt.ion()  
fig, ax = plt.subplots()  
image_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  
plt.axis('off')
plt.title("Yüz Tanıma Sistemi")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı.")
            break

        frame = cv2.flip(frame, 1) 
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Bilinmiyor"
            color = (0, 0, 255)

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)  

            if len(face_distances) > 0:
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                    name = kaydedilen_isimler[best_match]
                    color = (0, 255, 0)
                    if name not in kaydedilen_kisiler:
                        log_giris(name)
                        kaydedilen_kisiler.add(name)
                else:
                    # Bilinmeyen kişi algılandı
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    kaydet_bilinmeyen(frame, (left, top, right, bottom), name)
                    name = "Bilinmiyor"
                    color = (0, 0, 255)

            # Koordinatları geri ölçeklendir
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Yüzü çerçevele ve ismi yaz
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        image_display.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.pause(0.01)

        if not plt.fignum_exists(fig.number):  
            print("Pencere kapatıldı. Çıkılıyor.")
            break

except Exception as e:
    print(f"Hata oluştu: {e}")

cap.release()
plt.close('all')
