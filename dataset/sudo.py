import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input, Concatenate, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import ast
import gradio as gr

# ----------------------------
# 1. Data Integration Engine
# ----------------------------
class MedicalDataLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.disease_db = pd.DataFrame()

    def load_and_merge(self):
        # Load all CSV files
        data_files = {
            'symptoms': pd.read_csv(self.file_paths['symptoms']),
            'medications': pd.read_csv(self.file_paths['medications']),
            'diets': pd.read_csv(self.file_paths['diets']),
            'exercises': pd.read_csv(self.file_paths['exercises']),
            'contraindications': pd.read_csv(self.file_paths['contraindications'])
        }

        # Process list-type columns
        for key in data_files:
            data_files[key].iloc[:,1] = data_files[key].iloc[:,1].apply(ast.literal_eval)

        # Merge all data
        merged = data_files['symptoms']
        for key in ['medications', 'diets', 'exercises', 'contraindications']:
            merged = merged.merge(data_files[key], on='Disease')

        self.disease_db = merged
        return self

# ----------------------------
# 2. Advanced Preprocessing
# ----------------------------
class MedicalPreprocessor:
    def __init__(self, disease_db):
        self.disease_db = disease_db
        self.tokenizer = Tokenizer(oov_token='<UNK>')
        self.max_seq_length = 20

        # Initialize encoders
        self._prepare_encoders()

    def _prepare_encoders(self):
        # Symptom tokenization
        all_symptoms = [symptom for sublist in self.disease_db['Symptoms'] for symptom in sublist]
        self.tokenizer.fit_on_texts(all_symptoms)

        # Medication multi-label encoding
        all_meds = [med for sublist in self.disease_db['Medication'] for med in sublist]
        self.med_encoder = MultiLabelBinarizer()
        self.med_encoder.fit([all_meds])

        # Disease label encoding
        self.disease_encoder = LabelEncoder()
        self.disease_encoder.fit(self.disease_db['Disease'])

        # Demographic scaler
        self.scaler = MinMaxScaler()

    def preprocess_symptoms(self, symptoms):
        seq = self.tokenizer.texts_to_sequences([symptoms])
        return pad_sequences(seq, maxlen=self.max_seq_length, padding='post')

    def preprocess_demographics(self, demographics):
        return self.scaler.transform([demographics])

# ----------------------------
# 3. Hybrid LSTM-FNN Model
# ----------------------------
def build_medical_model(vocab_size, num_diseases, num_medications):
    # Symptom LSTM branch
    symptom_input = Input(shape=(None,), name='symptoms')
    x = Embedding(input_dim=vocab_size+1, output_dim=128)(symptom_input)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = LSTM(32)(x)
    x = Dense(64, activation='relu')(x)

    # Demographic FNN branch
    demo_input = Input(shape=(3,), name='demographics')
    y = Dense(64, activation='relu')(demo_input)
    y = Dense(32, activation='relu')(y)

    # Combined features
    combined = Concatenate()([x, y])
    combined = Dense(128, activation='relu')(combined)
    combined = Dropout(0.3)(combined)

    # Outputs
    disease_output = Dense(num_diseases, activation='softmax', name='disease')(combined)
    med_output = Dense(num_medications, activation='sigmoid', name='medications')(combined)

    return Model(inputs=[symptom_input, demo_input], outputs=[disease_output, med_output])

# ----------------------------
# 4. Training System
# ----------------------------
class MedicalTrainer:
    def __init__(self, file_paths):
        self.loader = MedicalDataLoader(file_paths).load_and_merge()
        self.preprocessor = MedicalPreprocessor(self.loader.disease_db)
        self.model = None

    def generate_synthetic_data(self, num_samples=50000):
        samples = []
        for _ in range(num_samples):
            disease_row = self.loader.disease_db.sample(1).iloc[0]
            symptoms = np.random.choice(
                disease_row['Symptoms'],
                size=np.random.randint(2, 5),
                replace=False
            )

            samples.append({
                'symptoms': symptoms,
                'age': np.random.randint(18, 90),
                'weight': np.random.uniform(40, 120),
                'gender': np.random.choice([0, 1]),
                'disease': disease_row['Disease'],
                'medications': disease_row['Medication']
            })

        return pd.DataFrame(samples)

    def train_model(self, epochs=30, batch_size=256):
        # Generate data
        df = self.generate_synthetic_data()

        # Preprocess symptoms
        X_symptoms = self.preprocessor.tokenizer.texts_to_sequences(df['symptoms'])
        X_symptoms = pad_sequences(X_symptoms, maxlen=self.preprocessor.max_seq_length)

        # Preprocess demographics
        demographics = df[['age', 'weight', 'gender']].values
        demographics = self.preprocessor.scaler.fit_transform(demographics)

        # Prepare outputs
        y_disease = self.preprocessor.disease_encoder.transform(df['disease'])
        y_meds = self.preprocessor.med_encoder.transform(df['medications'])

        # Build model
        self.model = build_medical_model(
            vocab_size=len(self.preprocessor.tokenizer.word_index),
            num_diseases=len(self.preprocessor.disease_encoder.classes_),
            num_medications=len(self.preprocessor.med_encoder.classes_)
        )

        # Compile with custom metrics
        self.model.compile(
            optimizer='adam',
            loss={
                'disease': 'sparse_categorical_crossentropy',
                'medications': 'binary_crossentropy'
            },
            metrics={
                'disease': 'accuracy',
                'medications': ['accuracy', tf.keras.metrics.AUC(name='auc')]
            }
        )

        # Train
        history = self.model.fit(
            x={'symptoms': X_symptoms, 'demographics': demographics},
            y={'disease': y_disease, 'medications': y_meds},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint('medical_model.h5', save_best_only=True)
            ]
        )
        return history

# ----------------------------
# 5. Safety-Aware Recommender
# ----------------------------
class MedicalAssistant:
    def __init__(self, trainer):
        self.preprocessor = trainer.preprocessor
        self.model = trainer.model
        self.disease_db = trainer.loader.disease_db

    def predict(self, symptoms, age, weight, gender):
        # Preprocess inputs
        symptom_seq = self.preprocessor.preprocess_symptoms(symptoms)
        demographics = self.preprocessor.preprocess_demographics([age, weight, gender])

        # Model prediction
        disease_probs, med_probs = self.model.predict({
            'symptoms': symptom_seq,
            'demographics': demographics
        })

        # Decode outputs
        disease_idx = np.argmax(disease_probs)
        disease = self.preprocessor.disease_encoder.inverse_transform([disease_idx])[0]

        medications = self.preprocessor.med_encoder.inverse_transform(
            (med_probs > 0.5).astype(int)
        )[0]

        # Get recommendations
        return {
            'diagnosis': disease,
            'medications': list(medications),
            'diet': self.disease_db[self.disease_db['Disease'] == disease]['Diet'].iloc[0],
            'exercise': self.disease_db[self.disease_db['Disease'] == disease]['Exercise'].iloc[0],
            'contraindications': self.disease_db[self.disease_db['Disease'] == disease]['Contraindications'].iloc[0]
        }

# ----------------------------
# 6. Gradio Interface
# ----------------------------
def launch_interface(trainer):
    assistant = MedicalAssistant(trainer)

    def predict_interface(symptoms, age, weight, gender):
        results = assistant.predict(
            symptoms.split(','),
            age,
            weight,
            0 if gender == 'Male' else 1
        )
        return {
            'Diagnosis': results['diagnosis'],
            'Recommended Medications': ', '.join(results['medications']),
            'Diet Plan': ', '.join(results['diet']),
            'Exercise Regimen': ', '.join(results['exercise']),
            'Contraindications': ', '.join(results['contraindications'])
        }

    return gr.Interface(
        fn=predict_interface,
        inputs=[
            gr.Textbox(label="Symptoms (comma-separated)"),
            gr.Number(label="Age"),
            gr.Number(label="Weight (kg)"),
            gr.Radio(["Male", "Female"], label="Gender")
        ],
        outputs=gr.JSON(label="Recommendations"),
        title="AI Medical Assistant",
        description="LSTM-FNN Hybrid Model for Medical Recommendations"
    )

# ----------------------------
# 7. Execution Pipeline
# ----------------------------
if __name__ == "__main__":
    # Configure file paths
    FILE_PATHS = {
        'symptoms': 'symptoms.csv',
        'medications': 'medications.csv',
        'diets': 'diets.csv',
        'exercises': 'exercises.csv',
        'contraindications': 'contraindications.csv'
    }

    # Initialize and train
    trainer = MedicalTrainer(FILE_PATHS)
    print("Training hybrid medical model...")
    history = trainer.train_model(epochs=30)

    # Launch interface
    print("Launching medical interface...")
    interface = launch_interface(trainer)
    interface.launch()
