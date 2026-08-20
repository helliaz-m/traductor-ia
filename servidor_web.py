import pickle
import os
from collections import deque
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

print("🧠 Cargando modelo de IA...", flush=True)
with open("modelo_cuerpo_entero.pkl", "rb") as f:
    modelo_ia = pickle.load(f)

# Memoria ultra-rápida pero estricta (solo 3 fotogramas)
historial = deque(maxlen=3)
ultima_traduccion = ""

# 🌟 NUEVO: Python ahora sirve la página web directamente
@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@socketio.on("traducir_frame")
def traducir_frame(data):
    global ultima_traduccion
    puntos = data.get("puntos", [])

    if not puntos: # Si no hay manos en la pantalla
        historial.clear()
        ultima_traduccion = ""
        emit("respuesta_traduccion", {"traduccion": ""})
        return

    try:
        vector = np.array(puntos).reshape(1, -1)
        probabilidades = modelo_ia.predict_proba(vector)[0]
        confianza = float(np.max(probabilidades))
        clase = str(modelo_ia.classes_[np.argmax(probabilidades)])

        # FILTRO 1: Ignorar si la IA no está al menos 75% segura
        if confianza >= 0.75:
            historial.append(clase)
        else:
            historial.append("")

        # FILTRO 2: Solo enviar la palabra si los últimos 3 fotogramas son EXACTAMENTE iguales
        if len(historial) == 3 and historial.count(historial[0]) == 3 and historial[0] != "":
            nueva_traduccion = historial[0]
            
            # Solo actualiza si es una palabra nueva, para no saturar la pantalla
            if nueva_traduccion != ultima_traduccion:
                ultima_traduccion = nueva_traduccion
                emit("respuesta_traduccion", {"traduccion": nueva_traduccion})
                
    except Exception as e:
        pass

if __name__ == "__main__":
    # host="0.0.0.0" permite que celulares en la misma red se conecten
    socketio.run(app, host="0.0.0.0", port=5000)