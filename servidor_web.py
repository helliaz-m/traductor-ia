import pickle
import os
from collections import deque
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=20)

print("🧠 Cargando modelo de IA...", flush=True)
with open("modelo_cuerpo_entero.pkl", "rb") as f:
    modelo_ia = pickle.load(f)

# 🔍 Diagnóstico: Imprime cuántas características espera el modelo
if hasattr(modelo_ia, "n_features_in_"):
    print(f"🎯 EL MODELO ESPERA EXACTAMENTE {modelo_ia.n_features_in_} PUNTOS", flush=True)

historial = deque(maxlen=2)
ultima_traduccion = ""

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

@socketio.on('connect')
def handle_connect():
    print("🟢 ¡Un usuario se ha conectado desde la página web!", flush=True)

@socketio.on('disconnect')
def handle_disconnect():
    print("🔴 Usuario desconectado.", flush=True)

@socketio.on("traducir_frame")
def traducir_frame(data):
    global ultima_traduccion
    puntos = data.get("puntos", [])

    if not puntos:
        historial.clear()
        ultima_traduccion = ""
        emit("respuesta_traduccion", {"traduccion": ""})
        return

    try:
        vector = np.array(puntos).reshape(1, -1)
        
        # Predicción de la IA
        probabilidades = modelo_ia.predict_proba(vector)[0]
        confianza = float(np.max(probabilidades))
        clase = str(modelo_ia.classes_[np.argmax(probabilidades)])

        # Imprime en Render los datos en tiempo real
        print(f"📥 Frame recibido ({len(puntos)} pts) -> Predicción: {clase} ({confianza*100:.1f}%)", flush=True)

        if confianza >= 0.35: # Umbral flexible para pruebas
            historial.append(clase)
        else:
            historial.append("")

        if len(historial) == 2 and historial.count(historial[0]) == 2 and historial[0] != "":
            nueva_traduccion = historial[0]
            print(f"✅ TRADUCCIÓN CONFIRMADA: {nueva_traduccion}", flush=True)
            ultima_traduccion = nueva_traduccion
            emit("respuesta_traduccion", {"traduccion": nueva_traduccion})

    except Exception as e:
        print(f"❌ ERROR EN PREDICCIÓN (Posible desfase de puntos): Esperaba {getattr(modelo_ia, 'n_features_in_', 'desconocido')} pts y recibió {len(puntos)} pts. Error: {e}", flush=True)
        emit("respuesta_traduccion", {"traduccion": f"Error dimensión ({len(puntos)} pts)"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
