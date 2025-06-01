# tracking_and_detection.py

import cv2
import random
import os
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import core.config as config
import core.api_client as api_client
import core.image_processing_utils as img_proc
from core.ocr_utils import executar_ocr_em_roi

# Inicialização dos modelos e tracker
try:
    modelo_carro = YOLO(config.MODELO_CARRO_PATH).to('cuda')
    modelo_placa = YOLO(config.MODELO_PLACA_PATH).to('cuda')
except Exception as e:
    print(f"Erro ao carregar modelos YOLO na GPU: {e}")
    print("Tentando carregar modelos na CPU...")
    try:
        modelo_carro = YOLO(config.MODELO_CARRO_PATH) # CPU
        modelo_placa = YOLO(config.MODELO_PLACA_PATH) # CPU
        print("Modelos YOLO carregados na CPU.")
    except Exception as e_cpu:
        print(f"Erro fatal ao carregar modelos YOLO na CPU: {e_cpu}")
        modelo_carro = None
        modelo_placa = None


tracker = DeepSort(max_age=config.DEEPSORT_MAX_AGE)
ocr_executor = ThreadPoolExecutor(max_workers=config.OCR_MAX_WORKERS)


placas_detectadas_recentemente = {}  # Cache de placas: {placa: (liberado, timestamp)}
veiculos_liberados_rastreados = set()  # IDs de tracks de veículos já liberados
placas_associadas_veiculo = {}  # track_id: placa_confirmada
contagem_placas_bloqueadas_por_veiculo = defaultdict(list) # track_id: [placa_ocr1, placa_ocr2,...]
veiculos_bloqueados_registrados = set()  #Veiculos bloqueados que já tiveram 5 capturas e já foram registrados na api como bloqueados
contador_frame_limpeza = 0 #Frames até a limpeza

def limpar_cache_expirado_placas():
    """Remove placas do cache que excederam o TTL."""
    agora = datetime.now()
    expiradas = [
        placa for placa, (_, timestamp) in placas_detectadas_recentemente.items()
        if agora - timestamp > timedelta(seconds=config.TTL_SEGUNDOS_CACHE_PLACA)
    ]
    for placa in expiradas:
        del placas_detectadas_recentemente[placa]

def limpar_variaveis_rastreamento(tracks):
    
    current_active_track_ids = {t.track_id for t in tracks}

    global veiculos_liberados_rastreados, placas_associadas_veiculo, contagem_placas_bloqueadas_por_veiculo,veiculos_bloqueados_registrados

    # Filtra veiculos_liberados_rastreados mantendo apenas os IDs ativos
    veiculos_liberados_rastreados = {tid for tid in veiculos_liberados_rastreados if tid in current_active_track_ids}

    # Filtra placas_associadas_veiculo mantendo apenas os IDs ativos
    placas_associadas_veiculo = {tid: placa for tid, placa in placas_associadas_veiculo.items() if tid in current_active_track_ids}

    # Filtra veiculos bloquados mantendo apenas os IDs ativos
    veiculos_bloqueados_registrados = {tid for tid in veiculos_bloqueados_registrados if tid in current_active_track_ids}

    # Filtra contagem_placas_bloqueadas_por_veiculo
    keys_to_remove_bloqueado = [tid for tid in contagem_placas_bloqueadas_por_veiculo if tid not in current_active_track_ids]
    for tid in keys_to_remove_bloqueado:
        del contagem_placas_bloqueadas_por_veiculo[tid]


def registrar_viculo(bbox_veiculo_abs,frame_para_desenho,placa,status):
    x1_car, y1_car, x2_car, y2_car = bbox_veiculo_abs 
    roi_veiculo_para_api = None 
    if frame_para_desenho is not None and x1_car < x2_car and y1_car < y2_car:
        y1_val = max(0, y1_car)
        y2_val = min(frame_para_desenho.shape[0], y2_car)
        x1_val = max(0, x1_car)
        x2_val = min(frame_para_desenho.shape[1], x2_car)
        if y1_val < y2_val and x1_val < x2_val:
            roi_veiculo_para_api = frame_para_desenho[y1_val:y2_val, x1_val:x2_val].copy()
            api_client.registrar_captura(placa, status,roi_veiculo_para_api,config.API_BASE_URL)



def processar_placa_identificada(placa_ocr, frame_para_desenho, bbox_veiculo_abs, track_id_veiculo):
    """
    Processa uma placa após o OCR: verifica na API, registra e atualiza o estado do veículo.
    bbox_veiculo_abs: (x1, y1, x2, y2) do veículo no frame original.
    """
    global placas_detectadas_recentemente, veiculos_liberados_rastreados
    global placas_associadas_veiculo, contagem_placas_bloqueadas_por_veiculo

    status_final_liberado = False
    x1_car, y1_car, _, _ = bbox_veiculo_abs # Usar para posicionar texto/retângulo

    # Verifica cache ou API
    if placa_ocr in placas_detectadas_recentemente:
        status_final_liberado, _ = placas_detectadas_recentemente[placa_ocr]
    else:
        timestamp_atual = datetime.now()
        status_final_liberado = api_client.verificar_placa_api(placa_ocr, config.API_BASE_URL)
        placas_detectadas_recentemente[placa_ocr] = (status_final_liberado, timestamp_atual)

    # Atualiza estado do veículo com base no status da placa
    if status_final_liberado:
        veiculos_liberados_rastreados.add(track_id_veiculo)
        registrar_viculo(bbox_veiculo_abs,frame_para_desenho,placa_ocr,"LIBERADO")
        placas_associadas_veiculo[track_id_veiculo] = placa_ocr
        if track_id_veiculo in contagem_placas_bloqueadas_por_veiculo: # Limpa contagens anteriores se agora está liberado
            del contagem_placas_bloqueadas_por_veiculo[track_id_veiculo]
    else:
        contagem_placas_bloqueadas_por_veiculo[track_id_veiculo].append(placa_ocr)


    status_texto = "LIBERADO" if status_final_liberado else "BLOQUEADO"
    cor_retangulo = (0, 255, 0) if status_final_liberado else (0, 0, 255)

    cv2.rectangle(frame_para_desenho, (x1_car, y1_car), (bbox_veiculo_abs[2], bbox_veiculo_abs[3]), cor_retangulo, 2)
    cv2.putText(frame_para_desenho, f"{placa_ocr} - {status_texto}", (x1_car, y1_car - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_retangulo, 2)
    
    #print(f"{datetime.now()} - Veículo ID {track_id_veiculo} - Placa: {placa_ocr} - Status: {status_texto}")


def detectar_e_rastrear(frame_original):
    """
    Função principal de detecção e rastreamento.
    Modifica o frame_original com desenhos e informações.
    """
    if modelo_carro is None or modelo_placa is None:
        cv2.putText(frame_original, "Modelos YOLO nao carregados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame_original

    global contador_frame_limpeza  # Declara que você vai usar a variável global
    contador_frame_limpeza += 1   # Incrementa a variável global

    
    limpar_cache_expirado_placas()
    roi_veiculo = frame_original
    frame_processamento = frame_original.copy()
    
    # 1. Detecção de veículos
    resultados_carros = modelo_carro(frame_processamento, conf=0.3, verbose=False) # Reduzir verbosidade
    
    deteccoes_para_tracker = []
    map_bbox_original_veiculo = {} 
    IDS_VEICULOS_INTERESSE = {2, 3, 5, 7} # carro, moto, onibus, caminhao

    altura_frame, largura_frame = frame_original.shape[:2]
    centro_frame_x = largura_frame // 2
    centro_frame_y = altura_frame // 2

    def distancia_do_centro(bbox):
        x1, y1, w, h = bbox
        cx = x1 + w // 2
        cy = y1 + h // 2
        return ((cx - centro_frame_x) ** 2 + (cy - centro_frame_y) ** 2) ** 0.5

    veiculos_detectados = []
    for res_carro in resultados_carros:
        for box in res_carro.boxes:
            try:
                cls = int(box.cls[0]) if hasattr(box.cls, '__getitem__') else int(box.cls)
                conf = float(box.conf[0]) if hasattr(box.conf, '__getitem__') else float(box.conf)

                if cls in IDS_VEICULOS_INTERESSE:
                    coords = box.xyxy[0] if hasattr(box.xyxy, '__getitem__') else box.xyxy
                    x1, y1, x2, y2 = map(int, coords)
                    largura = x2 - x1
                    altura = y2 - y1
                    area = largura * altura

                    if largura > 0 and altura > 0:
                        veiculos_detectados.append(((x1, y1, largura, altura), conf, area, cls))
            except Exception as e:
                print(f"Erro ao processar caixa de detecção: {e}")
                continue

    veiculos_detectados.sort(key=lambda x: (distancia_do_centro(x[0]), -x[2]))

    if veiculos_detectados: 
        bbox_data, conf, area, classe_id = veiculos_detectados[0]
        x1_veiculo, y1_veiculo, largura_veiculo, altura_veiculo = bbox_data
        x2_veiculo = x1_veiculo + largura_veiculo
        y2_veiculo = y1_veiculo + altura_veiculo

        x1_roi = max(0, x1_veiculo)
        y1_roi = max(0, y1_veiculo)
        x2_roi = min(largura_frame, x2_veiculo)
        y2_roi = min(altura_frame, y2_veiculo)

        if x1_roi < x2_roi and y1_roi < y2_roi:
            roi_veiculo = frame_original[y1_roi:y2_roi, x1_roi:x2_roi]
        else:
            roi_veiculo = None

        deteccoes_para_tracker.append((bbox_data, conf, "veiculo_generico", classe_id))
    else:
        roi_veiculo = None

    # 2. Atualização do rastreador DeepSort
    tracks = tracker.update_tracks(deteccoes_para_tracker, frame=frame_processamento)


    if contador_frame_limpeza >= config.LIMPEZA_VARIAVEIS_GLOBAIS:
        print("Limpeza de variaveis.")
        limpar_variaveis_rastreamento(tracks)
        contador_frame_limpeza=1

    
    ocr_tasks_ativas = [] 

    tracks_com_ocr_submetido_neste_frame = set()

    for track in tracks:
        #if not track.is_confirmed():
            #continue

        track_id = track.track_id
        x1_t, y1_t, x2_t, y2_t = map(int, track.to_ltrb()) 
        map_bbox_original_veiculo[track_id] = (x1_t, y1_t, x2_t, y2_t)

        # Se o veículo já foi liberado, apenas desenha o status e continua
        if track_id in veiculos_liberados_rastreados:
            placa = placas_associadas_veiculo.get(track_id, "N/A")
            cv2.rectangle(frame_original, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 0), 2)
            cv2.putText(frame_original, f"{placa} - LIBERADO (RASTREADO)", (x1_t, y1_t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            continue
        
        # Se o veículo foi consistentemente bloqueado, mostra status de bloqueado
        # e evita reprocessar OCR desnecessariamente por um tempo.
        # A lógica de "consistentemente bloqueado" (ex: 5 leituras da mesma placa bloqueada)
        if track_id in contagem_placas_bloqueadas_por_veiculo and len(contagem_placas_bloqueadas_por_veiculo[track_id]) > 5:
            contagem = Counter(contagem_placas_bloqueadas_por_veiculo[track_id])
            placa_mais_frequente_bloqueada, _ = contagem.most_common(1)[0]
            if track_id not in veiculos_bloqueados_registrados:
                veiculos_bloqueados_registrados.add(track_id)
                registrar_viculo(map_bbox_original_veiculo[track_id],frame_original,placa_mais_frequente_bloqueada,"BLOQUEADO")
            cv2.rectangle(frame_original, (x1_t, y1_t), (x2_t, y2_t), (0, 0, 255), 2)
            cv2.putText(frame_original, f"{placa_mais_frequente_bloqueada} - BLOQUEADO (RASTREADO)", (x1_t, y1_t - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            continue 
        

        if track_id in tracks_com_ocr_submetido_neste_frame:
            continue



        
        # Debug: nome base para salvar imagens
        nome_base_debug = f"track{track_id}_frame{random.randint(1000,9999)}"
        img_proc.salvar_imagem_debug("carro_original", frame_processamento, nome_base_debug, 0)
        img_proc.salvar_imagem_debug("carro_original2", roi_veiculo, nome_base_debug, 0)

        # Ajuste de brilho no ROI do veículo (opcional, pode ser feito no ROI da placa)
        brilho_roi_veiculo = img_proc.verificar_brilho(roi_veiculo)
        roi_veiculo_ajustado = roi_veiculo
        if brilho_roi_veiculo < config.BRILHO_MINIMO_NOTURNO:
            roi_veiculo_ajustado = img_proc.melhorar_visao_noturna(roi_veiculo)
            img_proc.salvar_imagem_debug("carro_noturno", roi_veiculo_ajustado, nome_base_debug, 0)
        elif brilho_roi_veiculo > config.BRILHO_MAXIMO_CLARO:
            roi_veiculo_ajustado = img_proc.melhorar_visao_clara(roi_veiculo)
            img_proc.salvar_imagem_debug("carro_claro", roi_veiculo_ajustado, nome_base_debug, 0)

        # Detecção de placas no ROI do veículo ajustado
        resultados_roi_placa = modelo_placa(roi_veiculo_ajustado, conf=0.25, verbose=False) # Ajustar conf para placas

        placa_idx_counter = 0
        for res_placa in resultados_roi_placa:
            for box_placa in res_placa.boxes:
                # Coordenadas da placa relativas ao ROI do veículo
                x1p_rel, y1p_rel, x2p_rel, y2p_rel = map(int, box_placa.xyxy[0])
                

                roi_placa_efetivo = roi_veiculo_ajustado[y1p_rel:y2p_rel, x1p_rel:x2p_rel]

                if roi_placa_efetivo.size == 0:
                    continue
                
                # img_proc.salvar_imagem_debug("placa_original_roi", roi_placa_efetivo, nome_base_debug, placa_idx_counter)

                # Submete a tarefa de OCR para o ThreadPoolExecutor
                # Passa o ROI da placa original, track_id, e o bbox original do veículo
                future = ocr_executor.submit(executar_ocr_em_roi, roi_placa_efetivo, nome_base_debug, placa_idx_counter)
                future.meta_info = {
                    "track_id": track_id,
                    "bbox_veiculo": (x1_t, y1_t, x2_t, y2_t) # bbox do veículo no frame original
                }
                ocr_tasks_ativas.append(future)
                tracks_com_ocr_submetido_neste_frame.add(track_id) # Marca que este track_id já teve OCR submetido
                placa_idx_counter +=1


    # 4. Coleta resultados das tarefas de OCR
    for future in as_completed(ocr_tasks_ativas):
        placas_encontradas_ocr = future.result() # Lista de strings de placas
        meta = future.meta_info
        track_id_veiculo = meta["track_id"]
        bbox_veiculo_original = meta["bbox_veiculo"]

        if placas_encontradas_ocr:

            placa_processar = placas_encontradas_ocr[0] 
            
            # Verifica se o veículo já não foi liberado por outra placa detectada anteriormente para o mesmo track
            if track_id_veiculo not in veiculos_liberados_rastreados:
                 processar_placa_identificada(placa_processar, frame_original, bbox_veiculo_original, track_id_veiculo)
            else:
                print(f"Nenhuma placa válida encontrada pelo OCR para track {track_id_veiculo}")

    return frame_original # Retorna o frame com os desenhos