# config.py

API_BASE_URL = "http://localhost:8000/api/"
TTL_SEGUNDOS_CACHE_PLACA = 600
LIMPEZA_VARIAVEIS_GLOBAIS = 2000
PROCESSAR_CADA_N_FRAMES = 1  # Não utilizado ativamente no loop principal do script original, mas mantido
PASTA_SAIDA_IMAGENS = "saida_deteccoes" # Pasta para salvar imagens de depuração/etapas
ARQUIVO_PLACAS = "placas_detectadas.txt"
# Configurações dos modelos (caminhos podem ser ajustados se necessário)
MODELO_CARRO_PATH = 'modelo_carro.pt'
MODELO_PLACA_PATH = 'modelo_placa.pt'

# Configurações do Tracker DeepSort
DEEPSORT_MAX_AGE = 5 # tempo de vida de um ID após sumir do frame

# Configurações OCR
OCR_ALLOWED_LIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Configurações da Câmera
#CAMERA_INDEX = "imagens" # Ou o caminho para um arquivo de vídeo
CAMERA_INDEX = 1 # Ou o caminho para um arquivo de vídeo

# Limites de pré-processamento
BRILHO_MINIMO_NOTURNO = 50
BRILHO_MAXIMO_CLARO = 150
ROI_PLACA_MIN_ALTURA = 30
ROI_PLACA_MIN_LARGURA = 100
ROI_PLACA_RESIZE_ESCALA_X = 1.5
ROI_PLACA_RESIZE_ESCALA_Y = 2.0
ROI_PLACA_FINAL_RESIZE_FX = 2.0
ROI_PLACA_FINAL_RESIZE_FY = 2.0

# Executor OCR
OCR_MAX_WORKERS = 6


# Confiança modelo
CONFIANCA_MODELO_AUTOMOVEL = 0.3