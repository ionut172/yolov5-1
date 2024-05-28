import torch
import cv2

# Încărcarea modelului YOLOv5 personalizat
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/test_run3/weights/best.pt')
    model.conf = 0.1  # Confidență minimă pentru detecția obiectelor
    print("Modelul a fost încărcat cu succes.")
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    exit()

# Încărcarea imaginii
imagine = cv2.imread('images/noi.jpeg')
if imagine is None:
    print("Eroare la încărcarea imaginii")
    exit()
else:
    print(f"Imaginea a fost încărcată cu succes. Dimensiuni: {imagine.shape}")

# Verificare dimensiuni imagine
print(f"Dimensiuni imagine: {imagine.shape}")

# Detectarea obiectelor
try:
    results = model(imagine)
    print("Detectarea obiectelor a fost realizată cu succes.")
except Exception as e:
    print(f"Eroare la detectarea obiectelor: {e}")
    exit()

# Obținerea rezultatelor sub formă de DataFrame
df = results.pandas().xyxy[0]

# Verificarea rezultatelor
if df.empty:
    print("Nu s-au detectat obiecte în imagine.")
else:
    print(f"Obiecte detectate: {len(df)}")
    print(df)

# Desenarea dreptunghiurilor de detecție pe imagine
for index, row in df.iterrows():
    x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']
    confidence = row['confidence']
    cv2.rectangle(imagine, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(imagine, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Salvarea imaginii cu detecții pentru verificare ulterioară
cv2.imwrite('results_with_detections.jpg', imagine)

# Afișarea imaginii cu detecții
cv2.imshow('YOLOv5 Object Detection', imagine)

# Așteaptă apăsarea unei taste pentru a închide fereastra
cv2.waitKey(0)
cv2.destroyAllWindows()
