# image_processing_utils.py

import cv2
import numpy as np
import os
import core.config as config

def corrigir_perspectiva(imagem):
    """Aplica uma correção de perspectiva simples na imagem."""
    if imagem is None or imagem.size == 0:
        return None
    h, w = imagem.shape[:2]
    # Pontos de origem (canto da imagem)
    src_pts = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    # Pontos de destino (um pouco para dentro para simular correção)
    # Estes valores podem precisar de ajuste fino dependendo da distorção real
    dst_pts = np.float32([[10, 10], [w - 11, 10], [10, h - 11], [w - 11, h - 11]])
    
    if h <= 20 or w <= 20: # Evita erro com imagens muito pequenas
        return imagem

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(imagem, M, (w, h))

def salvar_imagem_debug(etapa, imagem, nome_base, contador):
    """Salva uma imagem em uma etapa específica do processamento para debug."""
    if imagem is None or imagem.size == 0:
        print(f"[AVISO] Tentativa de salvar imagem vazia para: {nome_base}_{etapa}_{contador}")
        return

    if not os.path.exists(config.PASTA_SAIDA_IMAGENS):
        os.makedirs(config.PASTA_SAIDA_IMAGENS, exist_ok=True)
    caminho = os.path.join(config.PASTA_SAIDA_IMAGENS, f"{nome_base}_{etapa}_{contador}.jpg")
    cv2.imwrite(caminho, imagem)


def verificar_brilho(imagem):
    """Calcula o brilho médio de uma imagem."""
    if imagem is None or imagem.size == 0:
        return 0
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    brilho_medio = np.mean(cinza)
    return brilho_medio

def melhorar_visao_noturna(imagem):
    """Ajusta a imagem para simular melhoria em condições de pouca luz."""
    if imagem is None or imagem.size == 0:
        return None
    alpha = 1.8  # Contraste
    beta = 40    # Brilho
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

def melhorar_visao_clara(imagem):
    """Ajusta a imagem para simular melhoria em condições de muita luz."""
    if imagem is None or imagem.size == 0:
        return None
    alpha = 0.7  # Contraste
    beta = -30   # Brilho
    return cv2.convertScaleAbs(imagem, alpha=alpha, beta=beta)

def aplicar_clahe(imagem_gray):
    """Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization) à imagem em escala de cinza."""
    if imagem_gray is None or imagem_gray.size == 0:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(imagem_gray)


def preprocessar_roi_placa(roi_placa, nome_base_debug, contador_debug):
    """Realiza o pré-processamento completo em um ROI de placa."""
    imagem = roi_placa

    if imagem is None or imagem.size == 0:
        print(f"[ERRO] Imagem de ROI da placa vazia recebida ({nome_base_debug}_{contador_debug})")
        return None

    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    #salvar_imagem_debug("cinza", imagem_cinza, nome_base_debug, contador_debug)

    imagem_clahe = aplicar_clahe(imagem_cinza)
    #salvar_imagem_debug("clahe", imagem_clahe, nome_base_debug, contador_debug)
    
   
    altura_atual, largura_atual = imagem_clahe.shape[:2]
    escala_altura = config.ROI_PLACA_MIN_ALTURA / altura_atual if altura_atual < config.ROI_PLACA_MIN_ALTURA else 1.0
    escala_largura = config.ROI_PLACA_MIN_LARGURA / largura_atual if largura_atual < config.ROI_PLACA_MIN_LARGURA else 1.0

    # Usa o maior fator de escala necessário (o que garante que ambos os lados atinjam o mínimo)
    fator_escala = max(escala_altura, escala_largura)

    if fator_escala > 1.0:  # Só redimensiona se for necessário aumentar
        nova_largura = int(largura_atual * fator_escala)
        nova_altura = int(altura_atual * fator_escala)
        imagem_redimensionada = cv2.resize(imagem_clahe, (nova_largura, nova_altura), interpolation=cv2.INTER_CUBIC)
        #salvar_imagem_debug("redimensionada_sem_distorcao", imagem_redimensionada, nome_base_debug, contador_debug)
    else:
        imagem_redimensionada = imagem_clahe  # Mantém original se já atende o mínimo


    imagem_perspectiva = corrigir_perspectiva(imagem_redimensionada)
    if imagem_perspectiva is None: # Se a correção de perspectiva falhar
        imagem_perspectiva = imagem_redimensionada # Usa a imagem anterior
    #salvar_imagem_debug("perspectiva", imagem_perspectiva, nome_base_debug, contador_debug)
    
    # Binarização com Otsu
    _, imagem_binarizada = cv2.threshold(imagem_perspectiva, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #salvar_imagem_debug("binarizada", imagem_binarizada, nome_base_debug, contador_debug)


    kernel = np.ones((2,2),np.uint8) # Kernel pequeno
    imagem_limpa = cv2.morphologyEx(imagem_binarizada, cv2.MORPH_OPEN, kernel)
    #salvar_imagem_debug("07_morf_open", imagem_limpa, nome_base_debug, contador_debug)

 
    padding = 10
    imagem_final_ocr_com_padding = cv2.copyMakeBorder(imagem_limpa, padding, padding, padding, padding,
                                                 cv2.BORDER_CONSTANT, value=[255]) # Borda branca para imagem binarizada
    #salvar_imagem_debug("09_padding", imagem_final_ocr_com_padding, nome_base_debug, contador_debug)


    return imagem_final_ocr_com_padding