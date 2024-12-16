import cv2
from ultralytics import YOLO

model = YOLO('yolov5s.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()
else:
    print("Câmera conectada com sucesso.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("Detecção em Tempo Real", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
