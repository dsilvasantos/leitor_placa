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
# Se o EasyOCR tiver um logger específico, pode ser configurado aqui também.
# logging.getLogger("easyocr").setLevel(logging.WARNING)


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
'''
PASTA_IMAGENS_TESTE = "imagens_teste" 

def processar_imagens_pasta(pasta_imagens):
    if not os.path.exists(config.PASTA_SAIDA_IMAGENS):
        os.makedirs(config.PASTA_SAIDA_IMAGENS, exist_ok=True)

    if not os.path.exists(pasta_imagens):
        print(f"Pasta de imagens de teste '{pasta_imagens}' não encontrada.")
        return

    for nome_arquivo in os.listdir(pasta_imagens):
        if nome_arquivo.lower().endswith((".jpg", ".jpeg", ".png")):
            caminho_imagem = os.path.join(pasta_imagens, nome_arquivo)
            print(f"\\nProcessando imagem: {caminho_imagem}...")
            
            frame = cv2.imread(caminho_imagem)
            if frame is None:
                print(f"Não foi possível ler a imagem: {caminho_imagem}")
                continue

            # Chama a função de detecção e rastreamento
            frame_processado = detectar_e_rastrear(frame) # detectar_e_rastrear espera modificar o frame

            cv2.imshow('Imagem Processada', frame_processado)
            
            # Salva a imagem processada
            nome_saida = f"proc_{nome_arquivo}"
            caminho_saida = os.path.join(config.PASTA_SAIDA_IMAGENS, nome_saida)
            cv2.imwrite(caminho_saida, frame_processado)
            print(f"Imagem processada salva em: {caminho_saida}")

            if cv2.waitKey(0) & 0xFF == ord('q'): # Espera tecla 'q' para sair ou qualquer outra para continuar
                 break 
    cv2.destroyAllWindows()
    ocr_executor.shutdown(wait=True)
    print("Processamento de imagens da pasta concluído.")
'''

if __name__ == "__main__":
    # Para rodar com a câmera:
    main_loop_camera()
    
    # Para rodar com imagens de uma pasta (descomente a linha abaixo e comente main_loop_camera()):
    # processar_imagens_pasta(PASTA_IMAGENS_TESTE)