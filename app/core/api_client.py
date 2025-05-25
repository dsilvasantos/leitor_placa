# api_client.py

import requests
from datetime import datetime
import cv2 
import io  

def verificar_placa_api(placa, api_base_url):
    """Verifica o status de uma placa na API."""
    try:
        response = requests.get(f"{api_base_url}placas/{placa}")
        if response.status_code == 200:
            data = response.json()
            return data.get("liberado", False)
        else:
            print(f"Erro ao verificar placa {placa}: API retornou {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Exceção na verificação da placa {placa}: {e}")
        return False

def registrar_captura(
    placa: str, 
    status_liberado: str, 
    imagem_veiculo_roi, # Espera um array numpy como imagem
    api_url_base
):
    """
    Registra a captura de uma placa na API, onde a imagem do veículo é obrigatória.

    Args:
        placa (str): A placa do veículo.
        status_liberado (str): True se liberado, False se bloqueado.
        imagem_veiculo_roi (np.ndarray): Array NumPy representando a imagem do ROI do veículo.
                                         Deve ser uma imagem válida que cv2.imencode possa processar.
        api_url_base (str): URL base da API.

    Raises:
        ValueError: Se imagem_veiculo_roi não for fornecida ou for inválida.
        requests.RequestException: Para erros de rede ou HTTP da API.
    """
    if imagem_veiculo_roi is None or imagem_veiculo_roi.size == 0:
        raise ValueError("A imagem do veículo (imagem_veiculo_roi) é obrigatória e não foi fornecida ou está vazia.")

    endpoint_url = f"{api_url_base.rstrip('/')}/capturas"
    
    status_str = "LIBERADO" if status_liberado else "BLOQUEADO"
    
  
    files_to_send = None
    try:
        # Codificar a imagem para JPEG em memória
        is_success, buffer_img = cv2.imencode(".jpg", imagem_veiculo_roi)
        if not is_success:
            raise ValueError("Erro ao codificar a imagem do veículo para JPEG.")
        
        # A chave DEVE ser 'imagem' para coincidir com o servidor FastAPI
        files_to_send = {
            'imagem': ('veiculo.jpg', io.BytesIO(buffer_img.tobytes()), 'image/jpeg')
        }

          # Dados do formulário (sem a imagem)
        payload_data = {
        "placa": placa,
        "status": status_str
        }


        print(f"Enviando para {endpoint_url}: dados={payload_data}, arquivo={files_to_send['imagem'][0]}")

        response = requests.post(endpoint_url, data=payload_data, files=files_to_send, timeout=10) # Adicionado timeout

        response.raise_for_status() # Levanta uma exceção para status de erro HTTP (4xx ou 5xx)

        print(f"Captura para {placa} registrada com sucesso!")
        print(f"Resposta da API: {response.json()}")
        return response.json()

    except cv2.error as cv_err:
        print(f"Erro de OpenCV ao processar imagem para placa {placa}: {cv_err}")
    except requests.HTTPError as http_err:
        print(f"Erro HTTP ao registrar captura para {placa}: {http_err.response.status_code} - {http_err.response.text}")
    except requests.RequestException as req_err:
        print(f"Exceção de rede/conexão ao registrar captura para {placa}: {req_err}")
    except Exception as e:
        print(f"Ocorreu um erro inesperado no cliente para placa {placa}: {e}")
