import cv2
import re
import os
from ultralytics import YOLO
import easyocr
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import torch
import numpy as np
import random
import time


print(torch.__version__)
print(torch.cuda.is_available())  
print(torch.version.cuda)        
print(torch.cuda.get_device_name(0))  

API_BASE = "http://localhost:8000/api/"  # URL do serviço REST
PASTA_SAIDA = "saida"
#Classe temporal
# Configuração
placas_detectadas_recentemente = {}  # cache com TTL
TTL_SEGUNDOS = 120

ocr_executor = ThreadPoolExecutor(max_workers=4)  
score_minimo = 0.0


# Inicializações
modelo_carro = YOLO('modelo_carro.pt').to('cuda')
modelo_placa = YOLO('modelo_placa.pt').to('cuda')

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

def imagem_escura(imagem, limiar=50):
    """Verifica se a imagem tem baixa luminosidade."""
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(cinza)
    print("------------------------------  Brilho medio : " + str(brilho_medio))
    return brilho_medio < limiar

def imagem_clara(imagem, limiar=80):
    """Verifica se a imagem tem baixa luminosidade."""
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(cinza)
    print("------------------------------  Brilho medio : " + str(brilho_medio))
    return brilho_medio > limiar

def melhorar_visao_noturna(imagem):
    """Aplica equalização adaptativa para melhorar contraste em imagens escuras."""
    alpha = 1.8  
    beta = 40
    melhorada = cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)
    return melhorada

def melhorar_visao_clara(imagem):
    """Aplica equalização adaptativa para melhorar contraste em imagens claras."""
    alpha = -0.6  
    melhorada = cv2.convertScaleAbs(imagem, alpha=alpha, beta=0)
    return melhorada

def preprocessar_imagem_placa(roi_placa, nome_base, contador):

    imagem = cv2.cvtColor(roi_placa, cv2.COLOR_BGR2GRAY)

    # Redimensiona para melhorar OCR
    imagem = cv2.resize(imagem, None, fx=7, fy=7, interpolation=cv2.INTER_LINEAR)

    # Binarização
    _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
        if 7 == len(texto) :
            placa = texto
            break
        elif 3 == len(texto) :
            placa_moto[0] = texto
        elif 4  == len(texto) :
            placa_moto[1] = texto
        else :
            continue

    if  len(placa_moto[0]) == 3 and len(placa_moto[1]) == 4 :
        placa = placa_moto[0] + placa_moto[1]

    print("Texto OCR unido:", placa)

    # Tenta reconhecer padrões válidos
    matches = re.findall(r'[A-Z]{3}[0-9]{4}|[A-Z]{3}[0-9][A-Z][0-9]{2}', placa)
    placas_encontradas.extend(matches)

    return placas_encontradas

def processar_placa(placa, frame, x1, y1, x2, y2):
    status = "BLOQUEADO"

    if placa in placas_detectadas_recentemente:
        print(f"Placa {placa} já processada recentemente")
        liberado, agora = placas_detectadas_recentemente[placa]
    else :
        agora = datetime.now()
        liberado = verificar_placa(placa)
        registrar_captura(placa, liberado)
        placas_detectadas_recentemente[placa] = (liberado,agora)

    status = "LIBERADO" if liberado else "BLOQUEADO"

    cor = (0, 255, 0) if liberado else (0, 0, 255)
    cv2.rectangle(frame, (x1 - 25, y1 - 25), (x2 - 25, y2 -25), cor, 2)
    cv2.putText(frame, f"{placa} - {status}", (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor, 2)

    print(f"{agora} - Placa: {placa} - {status}")

def detectar_placas(frame):
    limpar_cache_placas()
    results = modelo_carro(frame)
    ocr_futures = []
    contador = 0
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            if int(cls) in [2, 3, 5, 7]:  # classes de veículos
                x1, y1, x2, y2 = map(int, box)
                if x1 >= x2 or y1 >= y2:
                    continue

                roi_carro = frame[y1:y2, x1:x2]
                nome = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=15))
                salvar_imagem("original carro", frame, nome, contador)

                # Detecta se a imagem está escura e aplica melhoria noturna se necessário
                esta_escuro = imagem_escura(roi_carro)
                score_minimo = 0.3 if esta_escuro else 0.5
                if(esta_escuro) :
                    print("Imagem escura detectada. Aplicando melhoria noturna...")
                    roi_carro = melhorar_visao_noturna(roi_carro)
                    salvar_imagem("melhorada carro", roi_carro, nome, contador)
                elif(imagem_clara(roi_carro)):
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

                        if roi_placa.size == 0:
                            continue

                        # Gera uma string de 7 caracteres aleatórios (com repetição)
                        
                        # OCR paralela: submit para execução
                        future = ocr_executor.submit(ocr_placa, roi_placa,nome,contador)
                        future.meta = (x1, y1, x2, y2, frame)  # anexa metadados
                        ocr_futures.append(future)
                        contador += 1


    # Processa os resultados paralelos
    for future in as_completed(ocr_futures):
        placas_detectadas = future.result()
        x1, y1, x2, y2, frame = future.meta
        for p in placas_detectadas:
            processar_placa(p, frame, x1, y1, x2, y2)

    return frame

# Cria uma janela redimensionável
cv2.namedWindow("Reconhecimento de Placas", cv2.WINDOW_NORMAL)
# Define o tamanho desejado da janela (por exemplo, 1280x720)
cv2.resizeWindow("Reconhecimento de Placas", 800, 600)

ultimo_tempo = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detectar_placas(frame)

    # Calcular e mostrar FPS
    tempo_atual = time.time()
    fps = 1 / (tempo_atual - ultimo_tempo)
    ultimo_tempo = tempo_atual
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Exibir imagem
    cv2.imshow("Reconhecimento de Placas", frame)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()