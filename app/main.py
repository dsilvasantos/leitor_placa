# main.py

import cv2
import logging
import os
import time

import core.config as config
from core.tracking_and_detection import detectar_e_rastrear, ocr_executor # Importa o executor para shutdown

# Configuração básica de logging para suprimir logs excessivos de bibliotecas
logging.basicConfig(level=logging.INFO)
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger("easyocr").setLevel(logging.WARNING)


def main_loop_camera():
    """Função principal para captura e processamento de vídeo da câmera."""
    
    if not os.path.exists(config.PASTA_SAIDA_IMAGENS):
        os.makedirs(config.PASTA_SAIDA_IMAGENS, exist_ok=True)
        print(f"Pasta de saída criada em: {config.PASTA_SAIDA_IMAGENS}")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Erro ao acessar a câmera no índice/caminho: {config.CAMERA_INDEX}")
        return

    cv2.namedWindow("Leitura de Placas - ALPR", cv2.WINDOW_NORMAL)
    # Definir um tamanho inicial, se desejado
    cv2.resizeWindow("Leitura de Placas - ALPR", 1280, 720) 

    frame_count = 0 # Para uso com PROCESSAR_CADA_N_FRAMES se necessário

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Não foi possível ler o frame da câmera ou o vídeo terminou.")
            break

        frame_count += 1
        # if config.PROCESSAR_CADA_N_FRAMES > 1 and frame_count % config.PROCESSAR_CADA_N_FRAMES != 0:
        #     # Pula o processamento deste frame, mas ainda exibe (ou exibe o anterior)
        #     if 'processed_frame' in locals(): # Exibe o último processado
        #          cv2.imshow("Leitura de Placas - ALPR", processed_frame)
        #     else: # Exibe o frame atual sem processamento
        #          cv2.imshow("Leitura de Placas - ALPR", frame)
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('q'): break
        #     continue


        # Chama a função de detecção e rastreamento que agora modifica o frame
        processed_frame = detectar_e_rastrear(frame)

        cv2.imshow('Leitura de Placas - ALPR', processed_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Tecla 'q' pressionada. Encerrando...")
            break
        elif key == ord('f'): # Tela cheia
            cv2.setWindowProperty("Leitura de Placas - ALPR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif key == ord('m'): # Modo normal/minimizado (ou tamanho padrão)
            cv2.setWindowProperty("Leitura de Placas - ALPR", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Leitura de Placas - ALPR", 1280, 720) # Redefine para tamanho padrão

    print("Liberando recursos...")
    cap.release()
    cv2.destroyAllWindows()
    print("Desligando OCR ThreadPoolExecutor...")
    ocr_executor.shutdown(wait=True)
    print("Aplicação encerrada.")

# Código comentado para processar imagens de uma pasta (manter como referência se necessário)


def processar_imagens_pasta():

    
    if not os.path.exists(config.PASTA_SAIDA_IMAGENS):
        os.makedirs(config.PASTA_SAIDA_IMAGENS, exist_ok=True)

    if not os.path.exists(config.CAMERA_INDEX):
        print(f"Pasta de imagens de teste '{config.CAMERA_INDEX}' não encontrada.")
        return

    for nome_arquivo in os.listdir(config.CAMERA_INDEX):
        if nome_arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                caminho_imagem = os.path.join(config.CAMERA_INDEX, nome_arquivo)
                print(f"\n --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ")
                print(f"\n Processando imagem: {caminho_imagem}...")

                frame = cv2.imread(caminho_imagem)
                if frame is None:
                    print(f"Não foi possível ler a imagem: {caminho_imagem}")
                    continue

                detectar_e_rastrear(frame)

                nome_saida = f"proc_{nome_arquivo}"
                caminho_saida = os.path.join(config.PASTA_SAIDA_IMAGENS, nome_saida)
                print(f"Imagem processada salva em: {caminho_saida}")
                time.sleep(1)
            except Exception as e:
                print(f"Erro ao processar a imagem {nome_arquivo}: {e}")
    ocr_executor.shutdown(wait=True)
    print("Processamento de imagens da pasta concluído.")


if __name__ == "__main__":
    # Para rodar com a câmera:
    main_loop_camera()
    
    # Para rodar com imagens de uma pasta (descomente a linha abaixo e comente main_loop_camera()):
    #processar_imagens_pasta()