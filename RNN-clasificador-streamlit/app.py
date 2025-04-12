# === Imports y Configuración Inicial ===
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences


# === Configuración de Streamlit ===
def configurar_pagina():
    st.set_page_config(page_title="Clasificador de Texto", layout="wide")
    st.markdown("""
        <style>
            .main .block-container { max-width: 95%; padding-left: 3rem; padding-right: 3rem; }
            pre { white-space: pre-wrap !important; word-break: break-word !important; }
        </style>
    """, unsafe_allow_html=True)
    st.title("🕸️ Clasificador de Texto con Deep Learning")


# === Carga de Datos y Tokenizer ===
def cargar_datos_tokenizer():
    df = pd.read_csv('data/df_total.csv')
    df['target'] = df['Type'].astype('category').cat.codes
    idx2label = dict(enumerate(df['Type'].astype('category').cat.categories))
    label2idx = {v: k for k, v in idx2label.items()}

    if os.path.exists('data/new_examples.csv'):
        df_new = pd.read_csv('data/new_examples.csv')
        df_new['target'] = df_new['Type'].map(label2idx)
        df = pd.concat([df, df_new], ignore_index=True)

    with open('tokenizer_23.pkl', 'rb') as f:
        tokenizer = pickle.load(f)

    return df, tokenizer, idx2label, label2idx

# === Preprocesamiento de Texto ===
def preprocesar_texto(df, tokenizer):
    sequences = tokenizer.texts_to_sequences(df['news'])
    data = pad_sequences(sequences)
    T = data.shape[1]
    return data, T


# === Construcción del Modelo ===
def construir_modelo(V, T, K, D=50):
    i = Input(shape=(T,))
    x = Embedding(V + 1, D)(i)
    x = LSTM(32, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(K)(x)
    modelo = Model(i, x)
    modelo.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(), metrics=['accuracy'])
    return modelo

# === Reentrenamiento del Modelo ===
def reentrenar_modelo(df, data, T, label2idx, tokenizer):
    st.sidebar.success("⏳ Entrenando modelo...")
    V = len(tokenizer.word_index)
    K = len(label2idx)

    modelo = construir_modelo(V, T, K)
    X_train, X_test, y_train, y_test = train_test_split(data, df['target'].values, test_size=0.3, random_state=42)
    history = modelo.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    modelo.save('modelo23.keras')
    with open('historial_entrenamiento_23.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    if os.path.exists('data/new_examples.csv'):
        os.remove('data/new_examples.csv')

    st.sidebar.success("✅ Reentrenamiento completo")
    st.session_state.retrain = False

# === Predicción ===
def predecir(modelo, tokenizer, T, texto):
    seq = tokenizer.texts_to_sequences([texto])
    padded = pad_sequences(seq, maxlen=T)
    pred = modelo.predict(padded)
    probs = tf.nn.softmax(pred[0]).numpy()
    return probs

# === Visualización de Probabilidades ===
def mostrar_probabilidades(probs, idx2label):
    df_probs = pd.DataFrame({
        'Clase': list(idx2label.values()),
        'Probabilidad': probs
    }).sort_values('Probabilidad', ascending=False)
    st.dataframe(df_probs, use_container_width=True)

    fig, ax = plt.subplots()
    colors = plt.cm.tab20.colors[:len(df_probs)]
    bar_colors = [colors[list(idx2label.values()).index(clase)] for clase in df_probs['Clase']]
    ax.barh(df_probs['Clase'], df_probs['Probabilidad'], color=bar_colors)
    ax.invert_yaxis()
    ax.grid(True, axis='x')
    st.pyplot(fig)

# === Visualización de Entrenamiento ===
def mostrar_historial_entrenamiento():
    try:
        with open('historial_entrenamiento_23.pkl', 'rb') as f:
            history = pickle.load(f)

        fig1, ax1 = plt.subplots()
        ax1.plot(history['loss'], label='Pérdida (entrenamiento)', color='blue')
        ax1.plot(history['val_loss'], label='Pérdida (validación)', color='orange')
        ax1.legend(); ax1.grid(); ax1.set_title("Pérdida"); st.pyplot(fig1)

        if 'accuracy' in history:
            fig2, ax2 = plt.subplots()
            ax2.plot(history['accuracy'], label='Precisión (entrenamiento)', color='red')
            ax2.plot(history['val_accuracy'], label='Precisión (validación)', color='black')
            ax2.legend(); ax2.grid(); ax2.set_title("Precisión"); st.pyplot(fig2)

    except FileNotFoundError:
        st.info("ℹ️ No se encontró el archivo `historial_entrenamiento_23.pkl`.")


# === Matriz de Confusión ===
def mostrar_matriz_confusion(df, modelo, tokenizer, T, idx2label):
    try:
        _, df_test = train_test_split(df, test_size=0.3, random_state=42)
        X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['news']), maxlen=T)
        y_test = df_test['target'].values

        y_pred = np.argmax(modelo.predict(X_test), axis=1)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(idx2label.values()))

        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"⚠️ Error al generar la matriz de confusión: {e}")

# === Distribución de Clases ===
def mostrar_distribucion_clases(df):
    fig, ax = plt.subplots()
    counts = df['Type'].value_counts()
    colors = plt.cm.tab20.colors[:len(counts)]
    counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_title('Cantidad de muestras por clase')
    st.pyplot(fig)

# === Evaluación del Modelo ===
def evaluar_modelo(df, modelo, tokenizer, T):
    try:
        _, df_test = train_test_split(df, test_size=0.3, random_state=42)
        X_test = pad_sequences(tokenizer.texts_to_sequences(df_test['news']), maxlen=T)
        y_test = df_test['target'].values

        loss, acc = modelo.evaluate(X_test, y_test, verbose=0)
        st.success(f"✅ Precisión del modelo: **{acc:.4f}**")
        st.info(f"📉 Pérdida del modelo: **{loss:.4f}**")
    except Exception as e:
        st.warning(f"⚠️ No se pudo evaluar el modelo: {e}")

# pie de página
def mostrar_firma_sidebar():
    st.sidebar.markdown("""
        <style>
            .firma-sidebar {
                position: fixed;
                bottom: 20px;
                left: 0;
                width: 20%;
                padding: 10px 15px;
                font-size: 0.8rem;
                border-radius: 10px;
                background-color: rgba(250, 250, 250, 0.9);
                z-index: 9999;
                text-align: left;
            }

            .firma-sidebar a {
                text-decoration: none;
                color: #333;
            }

            .firma-sidebar a:hover {
                color: #0077b5;
            }
        </style>

        <div class="firma-sidebar">
            Desarrollado por <strong>Mg. Luis Felipe Bustamante Narváez</strong><br>
            <a href="https://github.com/luizbn2" target="_blank">🐙 GitHub</a> · 
            <a href="https://www.linkedin.com/in/lfbn2" target="_blank">💼 LinkedIn</a>
        </div>
    """, unsafe_allow_html=True)

# === Interfaz Principal ===
def main():
    configurar_pagina()

    with st.sidebar:
        st.header("⚛️ Redes Neuronales Recurrentes")
        st.page_link("pages/visualizar_modelo.py", label="🧩 Estructura RNN")

        st.header("⚙️ Opciones Avanzadas")
        if st.button("🔁 Reentrenar modelo"):
            st.session_state.retrain = True

    st.info("🔄 Cargando datos y modelo...")
    df, tokenizer, idx2label, label2idx = cargar_datos_tokenizer()
    data, T = preprocesar_texto(df, tokenizer)

    if st.session_state.get("retrain"):
        reentrenar_modelo(df, data, T, label2idx, tokenizer)

    modelo = load_model('modelo23.keras', compile=False)
    modelo.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    # Entrada
    st.subheader("🔍 Clasificación de texto")
    metodo = st.radio("Selecciona el método de entrada:", ('Escribir texto', 'Subir archivo .txt'))
    texto = st.text_area("Texto a clasificar:") if metodo == 'Escribir texto' else \
        (st.file_uploader("Sube archivo .txt", type="txt").read().decode("utf-8") if st.file_uploader("Sube archivo .txt", type="txt") else "")

    if texto:
        probs = predecir(modelo, tokenizer, T, texto)
        pred_idx = np.argmax(probs)
        pred_label = idx2label[pred_idx]
        st.success(f"🔎 Predicción: **{pred_label}**")

        mostrar_probabilidades(probs, idx2label)

        st.subheader("📝 Corrección de etiqueta")
        correc_label = st.selectbox("Categoría correcta (si aplica):", list(idx2label.values()))
        if st.button("✅ Confirmar y aprender"):
            nuevo_df = pd.DataFrame({'news': [texto], 'Type': [correc_label]})
            nuevo_df['target'] = nuevo_df['Type'].map(label2idx)
            if os.path.exists('data/new_examples.csv'):
                df_ant = pd.read_csv('data/new_examples.csv')
                nuevo_df = pd.concat([df_ant, nuevo_df], ignore_index=True)
            nuevo_df.to_csv('data/new_examples.csv', index=False)
            st.success("🧠 Guardado para reentrenamiento")

    st.markdown("---")
    st.subheader("📈 Evolución del entrenamiento")
    mostrar_historial_entrenamiento()

    st.markdown("---")
    st.subheader("📊 Matriz de Confusión")
    mostrar_matriz_confusion(df, modelo, tokenizer, T, idx2label)

    st.markdown("---")
    st.subheader("📚 Distribución de Clases")
    mostrar_distribucion_clases(df)

    st.markdown("---")
    st.subheader("📌 Evaluación final del modelo")
    evaluar_modelo(df, modelo, tokenizer, T)


# === Lanzar App ===
if __name__ == '__main__':
    main()
    mostrar_firma_sidebar()
