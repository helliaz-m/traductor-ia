import pickle
import os
from collections import deque
from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
# 🔒 Sockets optimizados para la nube
socketio = SocketIO(app, cors_allowed_origins="*", ping_timeout=120, ping_interval=20)

print("🧠 Cargando modelo de IA...", flush=True)
with open("modelo_cuerpo_entero.pkl", "rb") as f:
    modelo_ia = pickle.load(f)

# 🎯 AFINACIÓN: Historial reducido (maxlen=2) para mayor velocidad
historial = deque(maxlen=2)
ultima_traduccion = ""

@app.route('/')
def index():
    return send_from_directory(os.getcwd(), 'index.html')

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
        probabilidades = modelo_ia.predict_proba(vector)[0]
        confianza = float(np.max(probabilidades))
        clase = str(modelo_ia.classes_[np.argmax(probabilidades)])

        # 🎯 AFINACIÓN: Umbral de confianza más tolerante (0.50 = 50%)
        if confianza >= 0.50:
            historial.append(clase)
        else:
            # 💡 Agrega un punto vacío si la confianza es baja para invalidar la secuencia
            historial.append("")

        # 🎯 AFINACIÓN: Solo requerimos 2 detecciones consecutivas idénticas
        if len(historial) == 2 and historial.count(historial[0]) == 2 and historial[0] != "":
            nueva_traduccion = historial[0]
            if nueva_traduccion != ultima_traduccion:
                print(f"✅ Traducción aceptada: {nueva_traduccion} (Confianza: {confianza:.2f})", flush=True)
                ultima_traduccion = nueva_traduccion
                emit("respuesta_traduccion", {"traduccion": nueva_traduccion})
                
    except Exception as e:
        # 🐛 Imprime cualquier error de dimensión para depurar
        print(f"🐛 Error de dimensión (Shape mismatch): Recibí {len(puntos)} puntos. Error: {e}", flush=True)
        pass

if __name__ == "__main__":
    # Lee automáticamente el puerto que Render asigna
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
