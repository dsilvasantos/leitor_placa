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

# Variáveis de estado (gerenciadas dentro deste módulo)
placas_detectadas_recentemente = {}  # Cache de placas: {placa: (liberado, timestamp)}
veiculos_liberados_rastreados = set()  # IDs de tracks de veículos já liberados
placas_associadas_veiculo = {}  # track_id: placa_confirmada
contagem_placas_bloqueadas_por_veiculo = defaultdict(list) # track_id: [placa_ocr1, placa_ocr2,...]
veiculos_bloqueados_registrados = set()  #Veiculos bloqueados que já tiveram 5 capturas e já foram registrados na api como bloqueados
contador_frame_limpeza = 0

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
    roi_veiculo_para_api = None # Inicializada como None
    if frame_para_desenho is not None and x1_car < x2_car and y1_car < y2_car:
        y1_val = max(0, y1_car)
        y2_val = min(frame_para_desenho.shape[0], y2_car)
        x1_val = max(0, x1_car)
        x2_val = min(frame_para_desenho.shape[1], x2_car)
        if y1_val < y2_val and x1_val < x2_val:
            # Esta linha DEVERIA produzir um array NumPy
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
    
    print(f"{datetime.now()} - Veículo ID {track_id_veiculo} - Placa: {placa_ocr} - Status: {status_texto}")


def detectar_e_rastrear(frame_original):
    """
    Função principal de detecção e rastreamento.
    Modifica o frame_original com desenhos e informações.
    """
    if modelo_carro is None or modelo_placa is None:
        cv2.putText(frame_original, "Modelos YOLO nao carregados", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame_original # Retorna o frame original se modelos não carregaram

    global contador_frame_limpeza  # Declara que você vai usar a variável global
    contador_frame_limpeza += 1   # Incrementa a variável global

    
    limpar_cache_expirado_placas()
   
    frame_processamento = frame_original.copy()
    
    # 1. Detecção de veículos
    resultados_carros = modelo_carro(frame_processamento, conf=config.CONFIANCA_MODELO_AUTOMOVEL, verbose=False) # Reduzir verbosidade
    
    map_bbox_original_veiculo = {} 
    roi_veiculo = None
    deteccoes_para_tracker = []
    IDS_VEICULOS_INTERESSE = {2, 3, 5, 7} # carro, moto, onibus, caminhao

    for res_carro in resultados_carros:
        melhor_deteccao_neste_res = None
        maior_confianca_neste_res = 0.1 

        for box in res_carro.boxes:
            try:
                cls = int(box.cls[0] if hasattr(box.cls, '__getitem__') else box.cls)
                conf = float(box.conf[0] if hasattr(box.conf, '__getitem__') else box.conf)
                if cls in IDS_VEICULOS_INTERESSE:
                    if conf > maior_confianca_neste_res:
                        # Esta é a melhor detecção de veículo encontrada até agora para este res_carro
                        maior_confianca_neste_res = conf
                        
                        coords = box.xyxy[0] if hasattr(box.xyxy, '__getitem__') and len(box.xyxy) > 0 else box.xyxy
                        x1, y1, x2, y2 = map(int, coords)

                        largura = x2 - x1
                        altura = y2 - y1

                        # Garante que a detecção original tem dimensões válidas
                        if largura > 0 and altura > 0:
                            melhor_deteccao_neste_res = ([x1, y1, largura, altura], conf, "veiculo_generico", cls)
            
            except Exception as e:
                print(f"Erro ao processar uma caixa de detecção: {e}")
                # Considere logar mais detalhes da 'box' se necessário para depuração
                continue # Pula para a próxima caixa


        if melhor_deteccao_neste_res is not None:
            bbox_data, confianca, _, classe_id = melhor_deteccao_neste_res
            x1_veiculo, y1_veiculo, largura_veiculo, altura_veiculo = bbox_data


            x2_veiculo = x1_veiculo + largura_veiculo
            y2_veiculo = y1_veiculo + altura_veiculo


            altura_frame, largura_frame = frame_original.shape[:2]


            x1_roi = max(0, x1_veiculo)
            y1_roi = max(0, y1_veiculo)
            x2_roi = min(largura_frame, x2_veiculo)
            y2_roi = min(altura_frame, y2_veiculo)


            if x1_roi < x2_roi and y1_roi < y2_roi:
                roi_veiculo = frame_original[y1_roi:y2_roi, x1_roi:x2_roi]
            else:
                roi_veiculo = None # Ou trate como preferir

            deteccoes_para_tracker.append(melhor_deteccao_neste_res) 


    # 2. Atualização do rastreador DeepSort
    tracks = tracker.update_tracks(deteccoes_para_tracker, frame=frame_processamento)

    # 3. Limpeza de variáveis DEPOIS de atualizar os tracks
    if contador_frame_limpeza >= config.LIMPEZA_VARIAVEIS_GLOBAIS: # Use >= para garantir que aconteça
        #print("\n Limpeza de variaveis.")
        limpar_variaveis_rastreamento(tracks) 
        contador_frame_limpeza = 1 

    ocr_tasks_ativas = [] 


    tracks_com_ocr_submetido_neste_frame = set()



    for track in tracks:
        print(f"Track ID: {track.track_id}, Hits: {track.hits}, Age: {track.age}, State: {track.state}, Confirmed: {track.is_confirmed()}")
        if not track.is_confirmed():  
            continue

        track_id = track.track_id
        x1_t, y1_t, x2_t, y2_t = map(int, track.to_ltrb()) # Coordenadas do tracker
        map_bbox_original_veiculo[track_id] = (x1_t, y1_t, x2_t, y2_t) 

        # Se o veículo já foi liberado ou bloqueado consistentemente, continua
        if track_id in veiculos_liberados_rastreados:
            # ... (seu código para desenhar veículo liberado) ...
            continue
        
        if track_id in contagem_placas_bloqueadas_por_veiculo and len(contagem_placas_bloqueadas_por_veiculo[track_id]) > 5:
            # ... (seu código para desenhar veículo bloqueado) ...
            continue
        
        nome_base_debug = f"track{track_id}_frame{random.randint(1000,9999)}"

        if track_id in tracks_com_ocr_submetido_neste_frame:
            continue

        # --- CORREÇÃO: Extrair ROI para o TRACK ATUAL ---
        altura_frame_original, largura_frame_original = frame_original.shape[:2]
        
        # "Clipar" (restringir) as coordenadas do track para os limites do frame_original
        x1_track_roi = max(0, x1_t)
        y1_track_roi = max(0, y1_t)
        x2_track_roi = min(largura_frame_original, x2_t)
        y2_track_roi = min(altura_frame_original, y2_t)

        # Verificar se a ROI do track resultante é válida (tem dimensões positivas)
        if not (x1_track_roi < x2_track_roi and y1_track_roi < y2_track_roi):
            # print(f"Track ID {track_id}: ROI inválida após clipping das coordenadas do track. Pulando.")
            continue 
        
        # Extrai a ROI específica para ESTE track
        roi_do_track_atual = frame_original[y1_track_roi:y2_track_roi, x1_track_roi:x2_track_roi]
        
        # Agora, use 'roi_do_track_atual' em vez de 'roi_veiculo'
        if roi_do_track_atual.size == 0: # Verificando a ROI correta
            # print(f"Track ID {track_id}: ROI do track atual tem tamanho 0. Pulando.")
            continue
        
        # Salve a imagem de debug usando a ROI correta, se necessário
        # img_proc.salvar_imagem_debug("carro_track_original", roi_do_track_atual, nome_base_debug, 0)
        
        # Use 'roi_do_track_atual' para os processamentos seguintes
        brilho_roi_veiculo = img_proc.verificar_brilho(roi_do_track_atual)
        roi_veiculo_ajustado = roi_do_track_atual # Começa com a ROI do track
        
        if brilho_roi_veiculo < config.BRILHO_MINIMO_NOTURNO:
            roi_veiculo_ajustado = img_proc.melhorar_visao_noturna(roi_do_track_atual) # Passa a ROI correta
            img_proc.salvar_imagem_debug("carro_noturno", roi_veiculo_ajustado, nome_base_debug, 0)
        elif brilho_roi_veiculo > config.BRILHO_MAXIMO_CLARO:
            roi_veiculo_ajustado = img_proc.melhorar_visao_clara(roi_do_track_atual) # Passa a ROI correta
            img_proc.salvar_imagem_debug("carro_claro", roi_veiculo_ajustado, nome_base_debug, 0)
        
        # Detecção de placas no ROI do veículo ajustado (que agora é derivado do track atual)
        resultados_roi_placa = modelo_placa(roi_veiculo_ajustado, conf=0.25, verbose=False)

        placa_idx_counter = 0
        for res_placa in resultados_roi_placa:
            for box_placa in res_placa.boxes:

                # Coordenadas da placa relativas ao ROI do veículo
                x1p_rel, y1p_rel, x2p_rel, y2p_rel = map(int, box_placa.xyxy[0])
                

                roi_placa_efetivo = roi_veiculo_ajustado[y1p_rel:y2p_rel, x1p_rel:x2p_rel]

                if roi_placa_efetivo.size == 0:
                    continue
                
                #img_proc.salvar_imagem_debug("placa_original_roi", roi_placa_efetivo, nome_base_debug, placa_idx_counter)

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

    return frame_original # Retorna o frame com os desenhos