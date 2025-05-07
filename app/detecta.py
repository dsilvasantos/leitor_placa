import cv2
import re
from ultralytics import YOLO
import easyocr
import requests
from datetime import datetime


API_BASE = "http://localhost:8000/api/"  # URL do serviço REST

# Inicializações
model = YOLO('yolov8n.pt')
reader = easyocr.Reader(['pt', 'en'])
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a câmera")
    exit()


def verificar_placa(placa):
    try:
        response = requests.get(API_BASE + "placas/" + placa)
        if response.status_code == 200:
            data = response.json()
            return data.get("liberado", False)
        else:
            return False
    except Exception as e:
        print(f"Erro na verificação da placa: {e}")
        return False

def registrar_captura(placa, status):
    try:
        payload = {"placa": placa, "status": status}
        response = requests.post(f"{API_BASE}capturas", json=payload)
        if response.status_code != 200:
            print(f"Erro ao registrar captura: {response.text}")
    except Exception as e:
        print(f"Exceção ao registrar captura: {e}")

def detectar_placas(frame):
    results = model(frame)
    placas = set()

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in [2, 3, 5, 7]:
                x1, y1, x2, y2 = map(int, box)
                if x1 < x2 and y1 < y2:
                    # Expande levemente os limites da caixa
                    x1_pad = max(0, x1 - 5)
                    y1_pad = max(0, y1 - 5)
                    x2_pad = min(frame.shape[1], x2 + 5)
                    y2_pad = min(frame.shape[0], y2 + 5)

                    roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                    # Reduz a ROI para focar só na parte interna da placa (evita molduras e fundo)
                    h, w = roi.shape[:2]
                    roi = roi[int(h*0.2):int(h*0.8), int(w*0.1):int(w*0.9)]

                    # Aumenta resolução
                    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

                    # Pré-processamento
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.bilateralFilter(roi_gray, 11, 17, 17)


                    resultados = reader.readtext(roi_gray)
                    for _, texto, _ in resultados:
                        texto = texto.upper().replace(" ", "").strip()
                        matches = re.findall(r'[A-Z]{3}-?[0-9][A-Z0-9][0-9]{2}', texto)
                        for placa in matches:
                            placa = placa.replace("-", "")
                            placas.add(placa)
                            
                            liberado = verificar_placa(placa)
                            status = "LIBERADO" if liberado else "BLOQUEADO"
                            cor = (0, 255, 0) if liberado else (0, 0, 255)

                            registrar_captura(placa, status)

                            print(f"{datetime.now()} - Placa: {placa} - {status}")

                            cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                            cv2.putText(frame, f"{placa} - {status}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, placa, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame, list(placas)

# Loop da câmera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, placas = detectar_placas(frame)
    cv2.imshow("Reconhecimento de Placas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
