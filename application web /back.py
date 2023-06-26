from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import joblib

app = Flask(__name__, static_folder='templates')

@app.route('/')
def index():
    return render_template('page.html')

if __name__ == '__main__':
    app.run()

# Chemin de téléchargement des modèles et des images
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/run_model', methods=['POST'])
def run_model():
    # Vérifier si les fichiers existent
    if 'model' not in request.files or 'image' not in request.files:
        return jsonify({'error': 'Fichiers manquants'}), 400

    model_file = request.files['model']
    image_file = request.files['image']

    # Vérifier si les fichiers sont valides
    if model_file.filename == '' or image_file.filename == '':
        return jsonify({'error': 'Fichier non valide'}), 400

    # Enregistrer les fichiers téléchargés
    model_filename = secure_filename(model_file.filename)
    image_filename = secure_filename(image_file.filename)
    model_filepath = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
    image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    model_file.save(model_filepath)
    image_file.save(image_filepath)

    # Charger le modèle
    model = joblib.load(model_filepath)

    # Effectuer la prédiction sur l'image
    # Ajoutez votre code de prétraitement de l'image ici si nécessaire
    prediction = model.predict(image_filepath)

    # Supprimer les fichiers téléchargés après utilisation
    os.remove(model_filepath)
    os.remove(image_filepath)

    # Retourner le résultat de la prédiction sous forme de réponse JSON
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run()
