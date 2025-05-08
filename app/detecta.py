import cv2
import re
from ultralytics import YOLO
import easyocr
import requests
from datetime import datetime
import threading


API_BASE = "http://localhost:8000/api/"  # URL do serviço REST

placas = dict()

#Classe temporal
def limpar_placas():
    global placas
    placas = dict()
    print("Placas limpas!")
    # Agenda a próxima execução daqui 60 segundos
    threading.Timer(60.0, limpar_placas).start()


# Inicializações
modelo_carro = YOLO('modelo_carro.pt')
modelo_placa = YOLO('modelo_placa.pt')

reader = easyocr.Reader(['pt'])
cap = cv2.VideoCapture(0)
limpar_placas()

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


import cv2
import re
from datetime import datetime

def detectar_placas(frame):
    results = modelo_carro(frame)

    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in [2, 3, 5, 7]:  # classes de veículos
                x1, y1, x2, y2 = map(int, box)

                if x1 < x2 and y1 < y2:
                    roi_carro = frame[y1:y2, x1:x2]

                    resultado_modelo_placa = modelo_placa(roi_carro)
                    for r_placa in resultado_modelo_placa:
                        for placa in r_placa.boxes.data.tolist():
                            x1p, y1p, x2p, y2p, score, class_id = placa
                            x1p, y1p, x2p, y2p = map(int, [x1p, y1p, x2p, y2p])

                            x1_abs = x1 + x1p
                            y1_abs = y1 + y1p
                            x2_abs = x1 + x2p
                            y2_abs = y1 + y2p

                            h, w = frame.shape[:2]
                            x1_abs = max(0, min(x1_abs, w))
                            x2_abs = max(0, min(x2_abs, w))
                            y1_abs = max(0, min(y1_abs, h))
                            y2_abs = max(0, min(y2_abs, h))

                            roi_placa = frame[y1_abs:y2_abs, x1_abs:x2_abs]
                            if roi_placa.size == 0:
                                continue

                            # Pré-processamento para OCR em ambiente claro
                            roi_placa_gray = cv2.cvtColor(roi_placa, cv2.COLOR_BGR2GRAY)

                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            roi_placa_gray = clahe.apply(roi_placa_gray)

                            roi_placa_gray = cv2.GaussianBlur(roi_placa_gray, (3, 3), 0)
                            roi_placa_gray = cv2.resize(roi_placa_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                            roi_placa_thresh = cv2.adaptiveThreshold(
                                roi_placa_gray, 255,
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY,
                                11, 2
                            )

                            resultados = reader.readtext(roi_placa_thresh)
                            if not resultados:
                                continue

                            for _, texto, _ in resultados:
                                texto = texto.upper().replace(" ", "").strip()
                                matches = re.findall(r'[A-Z]{3}-?[0-9][A-Z0-9][0-9]{2}', texto)

                                for placa in matches:
                                    placa = placa.replace("-", "")
                                    status = "BLOQUEADO"

                                    if placa not in placas:
                                        liberado = verificar_placa(placa)
                                        placas[placa] = liberado
                                        status = "LIBERADO" if liberado else "BLOQUEADO"
                                        registrar_captura(placa, status)
                                    else:
                                        print("Placa já reconhecida")

                                    cor = (0, 255, 0) if placas[placa] else (0, 0, 255)
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                                    cv2.putText(frame, f"{placa} - {status}", (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

                                    print(f"{datetime.now()} - Placa: {placa} - {status}")

    return frame


# Loop da câmera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detectar_placas(frame)

    # Cria uma janela redimensionável
    cv2.namedWindow("Reconhecimento de Placas", cv2.WINDOW_NORMAL)
    # Define o tamanho desejado da janela (por exemplo, 1280x720)
    cv2.resizeWindow("Reconhecimento de Placas", 1280, 720)

    frame_ampliado = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)

    # Exibe o frame
    cv2.imshow("Reconhecimento de Placas", frame_ampliado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
