from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

# Cargar el modelo previamente entrenado y guardado
model = tf.keras.models.load_model('relaks_model_v01.h5')

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, I'm aliveee!"

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar si el cuerpo de la solicitud está vacío
        if not request.is_json:
            return jsonify({"error": "Por favor, envíe los datos en formato JSON"}), 400
        
        # Obtener los datos enviados en formato JSON
        data = request.get_json()

        # Validar que todos los campos necesarios están presentes
        required_fields = ['HR', 'SpO2', 'Sleep_Minutes', 'Puntaje_STAI', 'Preferencia_Tecnica']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Falta el campo '{field}' en la solicitud"}), 400

        # Extraer los valores (HR, SpO2, Sleep_Minutes, Puntaje_STAI, Preferencia_Tecnica)
        nuevos_datos = np.array([[data['HR'], data['SpO2'], data['Sleep_Minutes'], data['Puntaje_STAI'], data['Preferencia_Tecnica']]], dtype=float)

        # Hacer la predicción
        prediccion = model.predict(nuevos_datos)

        # Redondear la predicción al número entero más cercano
        prediccion_redondeada = round(prediccion[0][0])

        # Devolver la predicción como JSON
        return jsonify({
            'Tecnica_Recomendada_ID': int(prediccion_redondeada)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
