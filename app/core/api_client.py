# api_client.py

import requests
from datetime import datetime

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

def registrar_captura_api(placa, status_liberado, api_base_url):
    """Registra a captura de uma placa na API."""
    try:
        payload = {"placa": placa, "status": "LIBERADO" if status_liberado else "BLOQUEADO"}
        response = requests.post(f"{api_base_url}capturas", json=payload)
        if response.status_code != 200:
            print(f"Erro ao registrar captura para {placa}: {response.status_code} - {response.text}")
    except requests.RequestException as e:
        print(f"Exceção ao registrar captura para {placa}: {e}")