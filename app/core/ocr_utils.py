# ocr_utils.py

import easyocr
import re
import core.config as config
from core.image_processing_utils import preprocessar_roi_placa #, salvar_imagem_debug

ARQUIVO_PLACAS = config.ARQUIVO_PLACAS
open(ARQUIVO_PLACAS, 'w').close()

# Inicializa o reader do EasyOCR uma vez quando o módulo é carregado
try:
    print("Inicializando EasyOCR Reader...")
    reader_ocr = easyocr.Reader(['pt'], gpu=True, detect_network="craft", verbose=False)
    print("EasyOCR Reader inicializado.")
except Exception as e:
    print(f"Erro ao inicializar EasyOCR: {e}")
    print("Tentando inicializar EasyOCR com GPU=False.")
    try:
        reader_ocr = easyocr.Reader(['pt'], gpu=False, detect_network="craft", verbose=False)
        print("EasyOCR Reader inicializado com CPU.")
    except Exception as e_cpu:
        print(f"Erro fatal ao inicializar EasyOCR com CPU: {e_cpu}")
        reader_ocr = None


def substituir_caracteres_similares_letras(texto):
    """Corrige letras que podem ser confundidas com números (ex: O -> 0)."""
    substituicoes = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}
    texto_corrigido = list(texto)
    # Aplica somente nos 3 primeiros caracteres (letras)
    for i in range(min(3, len(texto_corrigido))):
        if texto_corrigido[i] in substituicoes:
            texto_corrigido[i] = substituicoes[texto_corrigido[i]]
    return ''.join(texto_corrigido)

def substituir_caracteres_similares_numeros(texto):
    """Corrige números que podem ser confundidas com letras (ex: O -> 0)."""
    substituicoes = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}
    texto_corrigido = list(texto)

    indices_numeros_trad = [3, 4, 5, 6]
    indices_numeros_merco = [3, 5, 6]

    # Heurística simples para decidir qual padrão usar
    # Se o 4º caractere (índice 3) for letra após correção inicial, pode ser Mercosul
    potencial_mercosul = len(texto_corrigido) == 7 and texto_corrigido[4].isalpha()

    indices_para_corrigir = indices_numeros_merco if potencial_mercosul else indices_numeros_trad

    for i in indices_para_corrigir:
        if i < len(texto_corrigido) and texto_corrigido[i] in substituicoes:
            texto_corrigido[i] = substituicoes[texto_corrigido[i]]
    return ''.join(texto_corrigido)


def validar_e_formatar_placa(texto_ocr):
    """
    Valida o texto da placa usando regex para padrões brasileiros (tradicional e Mercosul)
    e aplica correções de caracteres.
    """
    texto = texto_ocr.upper().replace(" ", "").replace("-", "").replace(".", "").strip()

    if len(texto) != 7:
        return None


    # Padrões de placa: LLLNNNN (tradicional) e LLLNLNN (Mercosul)
    padrao_tradicional = r'^[A-Z]{3}[0-9]{4}$'
    padrao_mercosul = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'

    # Se o texto original já bate com o padrão (sem correções)
    if re.fullmatch(padrao_tradicional, texto) or \
       re.fullmatch(padrao_mercosul, texto):
        return texto

    # Tentativa de correção inicial (letras)
    texto_corrigido_letras = substituir_caracteres_similares_letras(texto)
    
    # Se o texto original não bate tenta com a correção de letras
    if re.fullmatch(padrao_tradicional, texto_corrigido_letras) or \
       re.fullmatch(padrao_mercosul, texto_corrigido_letras):
        return texto_corrigido_letras

    # Tentativa de correção  numeros (números)
    texto_corrigido_numeros = substituir_caracteres_similares_numeros(texto)

    #Se o tecto original não bate e de letras idem tenta com a correção de números
    if re.fullmatch(padrao_tradicional, texto_corrigido_numeros) or \
       re.fullmatch(padrao_mercosul, texto_corrigido_numeros):
        return texto_corrigido_numeros
    

    #Se o texto original não bate, letras e núemros idem tenta com a correção de letras e números
    texto_corrigido_numeros_e_letras = substituir_caracteres_similares_numeros(texto_corrigido_letras)

    if re.fullmatch(padrao_tradicional, texto_corrigido_numeros_e_letras) or \
       re.fullmatch(padrao_mercosul, texto_corrigido_numeros_e_letras):
        return texto_corrigido_numeros_e_letras
    
    
    return None


def executar_ocr_em_roi(roi_placa_original, nome_base_debug, contador_debug):
    """
    Executa o OCR em uma Região de Interesse (ROI) da placa.
    Retorna uma lista de placas validadas encontradas.
    """
    if reader_ocr is None:
        print("[ERRO OCR] EasyOCR Reader não foi inicializado.")
        return []
        
    if roi_placa_original is None or roi_placa_original.size == 0:
        return []

    imagem_placa_preprocessada = preprocessar_roi_placa(roi_placa_original, nome_base_debug, contador_debug)

    if imagem_placa_preprocessada is None or imagem_placa_preprocessada.size == 0:
        return []

    placas_validadas = []
    try:
        resultados_ocr = reader_ocr.readtext(imagem_placa_preprocessada, allowlist=config.OCR_ALLOWED_LIST, paragraph=False)
        
        if not resultados_ocr: 
            return placas_validadas

        placa = ''
        placa_moto = ["",""]
        
        for _, texto, _ in resultados_ocr:
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

        placa_formatada = validar_e_formatar_placa(placa)
        if placa_formatada:
            placas_validadas.append(placa_formatada)
    
    
    except Exception as e:
        print(f"[ERRO OCR] Exceção durante o readtext ou processamento: {e}")

    
    with open(ARQUIVO_PLACAS, 'a') as f:
        for p in placas_validadas:
            f.write(f"{nome_base_debug}_{contador_debug}: {p}\n")
    
    return placas_validadas