import torch
import cv2

# Încărcarea modelului YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Deschiderea camerei live (folosim 0 pentru camera implicită)
cap = cv2.VideoCapture(0)

# Verificăm dacă camera a fost deschisă cu succes
if not cap.isOpened():
    print("Eroare la deschiderea camerei")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detectarea obiectelor
    results = model(frame)

    # Obținerea rezultatelor sub formă de DataFrame
    df = results.pandas().xyxy[0]

    # Desenarea dreptunghiurilor de detecție pe frame
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afișarea frame-ului cu detecții
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Apăsăm 'q' pentru a ieși din buclă
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
