import cv2
import re
import os
from ultralytics import YOLO
import easyocr
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import numpy as np
import random
import time
from deep_sort_realtime.deepsort_tracker import DeepSort


API_BASE = "http://localhost:8000/api/"  # URL do serviço REST
PASTA_SAIDA = "saida"
placas_detectadas_recentemente = {}  # cache com TTL
TTL_SEGUNDOS = 120
PROCESSAR_CADA_N_FRAMES = 1
contador_frame = 0
ocr_executor = ThreadPoolExecutor(max_workers=6)  

# Inicializações
modelo_carro = YOLO('modelo_carro.pt').to('cuda')
modelo_placa = YOLO('modelo_placa.pt').to('cuda')

tracker = DeepSort(max_age=10)  # tempo de vida de um ID após sumir do frame
veiculos_liberados = set()  # IDs de veículos já liberados
placas_por_veiculo = {}  # track_id: placa

logging.getLogger("ultralytics").setLevel(logging.WARNING)
reader = easyocr.Reader(['pt'], gpu=True, detect_network="craft", verbose=False)
cap = cv2.VideoCapture(2)

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
        payload = {"placa": placa, "status": "LIBERADO" if status else "BLOQUEADO"}
        response = requests.post(f"{API_BASE}capturas", json=payload)
        if response.status_code != 200:
            print(f"Erro ao registrar captura: {response.text}")
    except Exception as e:
        print(f"Exceção ao registrar captura: {e}")

def limpar_cache_placas():
    agora = datetime.now()
    expiradas = [p for p, (_, t) in placas_detectadas_recentemente.items() if agora - t > timedelta(seconds=TTL_SEGUNDOS)]
    for p in expiradas:
        del placas_detectadas_recentemente[p]

def corrigir_perspectiva(imagem):
    h, w = imagem.shape[:2]
    src_pts = np.float32([[0,0], [w,0], [0,h], [w,h]])
    dst_pts = np.float32([[10,10], [w-10,10], [10,h-10], [w-10,h-10]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(imagem, M, (w,h))

def salvar_imagem(etapa, imagem, nome_base, contador):
    caminho = os.path.join(PASTA_SAIDA, f"{nome_base}_{etapa}_{contador}.jpg")
    cv2.imwrite(caminho, imagem)

def verificar_brilho(imagem,nome_base):
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(cinza)
    print("Imagem: " + nome_base + " ------------------------------  Brilho medio : " + str(brilho_medio))
    return brilho_medio

def melhorar_visao_noturna(imagem):
    alpha = 1.8  
    beta = 40
    melhorada = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    return melhorada

def melhorar_visao_clara(imagem):
    alpha = 0.7  
    beta = -30
    melhorada = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    return melhorada

def preprocessar_imagem_placa(roi_placa, nome_base, contador):
    imagem = roi_placa
    h, w = imagem.shape[:2]

    print(nome_base)
    print(h,w)

    if h < 30 or w < 100:
        print("Imagem pequena. Aplicando upscale inicial...")
        escala_x = 1.5
        escala_y = 2.0
        nova_largura = int(w * escala_x)
        nova_altura = int(h * escala_y)
        imagem = cv2.resize(imagem, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)
        salvar_imagem("upscaled", imagem, nome_base, contador)
    h, w = imagem.shape[:2]

    print(nome_base)
    print(h,w)
    
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    salvar_imagem("gray", imagem, nome_base, contador)

    if imagem.shape[0] < 100 or imagem.shape[1] < 200:
        imagem = cv2.resize(imagem, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        salvar_imagem("resize", imagem, nome_base, contador)

    _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    salvar_imagem("threshold", imagem, nome_base, contador)

    return imagem

def ocr_placa(imagem_placa, nome_base, contador):
    placas_encontradas = []

    imagem_pre = preprocessar_imagem_placa(imagem_placa, nome_base, contador)
    resultados = reader.readtext(imagem_pre, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    placa = ''
    placa_moto = ["",""]
    for _, texto, _ in resultados:
        print("Texto : " + texto)
        texto = texto.upper().replace(" ", "").strip()
        if 7 == len(texto):
            placa = texto
            break
        elif 3 == len(texto):
            placa_moto[0] = texto
        elif 4 == len(texto):
            placa_moto[1] = texto
        else:
            continue

    if len(placa_moto[0]) == 3 and len(placa_moto[1]) == 4:
        placa = placa_moto[0] + placa_moto[1]

    print("Texto OCR unido:", placa)

    matches = re.findall(r'[A-Z]{3}[0-9]{4}|[A-Z]{3}[0-9][A-Z][0-9]{2}', placa)
    placas_encontradas.extend(matches)

    return placas_encontradas

def processar_placa(placa, frame, x1, y1, x2, y2, track_id):
    status = "BLOQUEADO"

    if placa in placas_detectadas_recentemente:
        print(f"Placa {placa} já processada recentemente")
        liberado, agora = placas_detectadas_recentemente[placa]
    else:
        agora = datetime.now()
        liberado = verificar_placa(placa)
        registrar_captura(placa, liberado)
        placas_detectadas_recentemente[placa] = (liberado, agora)

    if liberado:
        veiculos_liberados.add(track_id)
        placas_por_veiculo[track_id] = placa

    status = "LIBERADO" if liberado else "BLOQUEADO"

    cor = (0, 255, 0) if liberado else (0, 0, 255)
    cv2.rectangle(frame, (x1 - 25, y1 - 25), (x2 - 25, y2 - 25), cor, 2)
    cv2.putText(frame, f"{placa} - {status}", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    print(f"{agora} - Placa: {placa} - {status}")

def detectar_placas(frame):
    limpar_cache_placas()
    results = modelo_carro(frame, conf=0.7)
    ocr_futures = []
    contador = 0

    carros_detectados = []

    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            if int(cls) in [2, 3, 5, 7]:  # classes de veículos
                x1, y1, x2, y2 = map(int, box)
                carros_detectados.append(([x1, y1, x2 - x1, y2 - y1], conf.item(), "carro"))

    # Atualiza o rastreamento fora do loop de detecção
    tracks = tracker.update_tracks(carros_detectados, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        if track_id in veiculos_liberados:
            # Mostra novamente o quadrado verde com a placa e status
            placa = placas_por_veiculo.get(track_id, "")
            cv2.rectangle(frame, (x1 - 25, y1 - 25), (x2 - 25, y2 - 25), (0, 255, 0), 2)
            cv2.putText(frame, f"{placa} - LIBERADO", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            continue

        roi_carro = frame[y1:y2, x1:x2]
        if roi_carro.size == 0:
            print(f"ROI inválido para track_id {track_id}. Pulando...")
            continue
        
        nome = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=15))
        salvar_imagem("original carro", frame, nome, contador)

        score_minimo = 0.3
        brilho = verificar_brilho(roi_carro, nome)

        if brilho < 50:
            print("Imagem escura detectada. Aplicando melhoria noturna...")
            roi_carro = melhorar_visao_noturna(roi_carro)
            salvar_imagem("melhorada carro", roi_carro, nome, contador)
        elif brilho > 80:
            print("Imagem clara detectada. Aplicando melhoria de claridade...")
            roi_carro = melhorar_visao_clara(roi_carro)
            salvar_imagem("melhorada carro", roi_carro, nome, contador)

        resultado_modelo_placa = modelo_placa(roi_carro)

        for r_placa in resultado_modelo_placa:
            for placa in r_placa.boxes.data.tolist():
                x1p, y1p, x2p, y2p, score, _ = placa
                if score < score_minimo:
                    continue

                x1p, y1p, x2p, y2p = map(int, [x1p, y1p, x2p, y2p])
                x1_abs = max(0, min(x1 + x1p, frame.shape[1]))
                y1_abs = max(0, min(y1 + y1p, frame.shape[0]))
                x2_abs = max(0, min(x1 + x2p, frame.shape[1]))
                y2_abs = max(0, min(y1 + y2p, frame.shape[0]))

                roi_placa = frame[y1_abs:y2_abs, x1_abs:x2_abs]

                salvar_imagem("original placa", roi_placa, nome, contador)

                future = ocr_executor.submit(ocr_placa, roi_placa, nome, contador)
                future.meta = (frame, x1_abs, y1_abs, x2_abs, y2_abs, track_id)
                ocr_futures.append(future)

        contador += 1

    for future in as_completed(ocr_futures):
        placas = future.result()
        frame, x1_abs, y1_abs, x2_abs, y2_abs, track_id = future.meta

        for placa in placas:
            processar_placa(placa, frame, x1, y1, x2, y2, track_id)

def main():
    global contador_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contador_frame += 1
        if contador_frame % PROCESSAR_CADA_N_FRAMES != 0:
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        detectar_placas(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()