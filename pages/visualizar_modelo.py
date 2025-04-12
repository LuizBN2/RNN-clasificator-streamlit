# === Imports ===
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
import io
from PIL import Image
from contextlib import redirect_stdout
import pydot

# === Configuración de Página ===
def configurar_pagina():
    st.set_page_config(page_title="Capas del Modelo", layout="wide")
    st.markdown("""
        <style>
            .main .block-container {
                max-width: 95%;
                padding-left: 3rem;
                padding-right: 3rem;
            }
            pre {
                white-space: pre-wrap !important;
                word-break: break-word !important;
            }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.page_link("app.py", label="🏠 Página Principal")
    st.title("🔬 Visualización de la Red Neuronal")

# === Cargar Modelo ===
def cargar_modelo(ruta):
    st.info("📥 Cargando modelo...")
    return load_model(ruta, compile=False)

# === Mostrar Resumen del Modelo ===
def mostrar_resumen_modelo(modelo):
    st.subheader("📋 Resumen del Modelo")
    summary_buffer = io.StringIO()
    with redirect_stdout(summary_buffer):
        modelo.summary()
    st.code(summary_buffer.getvalue(), language='text')

# === Generar archivo DOT personalizado ===
def generar_dot_con_tablas(model, output_path):
    layer_colors = {
        "Embedding": "#dcedc8",
        "LSTM": "#ffccbc",
        "GlobalMaxPooling1D": "#ffe082",
        "Dense": "#bbdefb",
        "InputLayer": "#f0f0f0"
    }

    def get_shape_safe(tensor):
        try:
            return str(tensor.shape)
        except:
            return "?"

    with open(output_path, "w") as f:
        f.write("digraph G {\n")
        f.write("    rankdir=TB;\n")
        f.write("    concentrate=true;\n")
        f.write("    dpi=200;\n")
        f.write("    splines=ortho;\n")
        f.write("    node [shape=plaintext fontname=Helvetica];\n\n")

        for i, layer in enumerate(model.layers):
            name = layer.name
            tipo = layer.__class__.__name__
            node_id = f"layer_{i}"

            try:
                input_shape = str(layer.input_shape)
            except:
                input_shape = get_shape_safe(layer.input) if hasattr(layer, "input") else "?"

            try:
                output_shape = str(layer.output_shape)
            except:
                output_shape = get_shape_safe(layer.output) if hasattr(layer, "output") else "?"

            color = layer_colors.get(tipo, "#eeeeee")

            label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6" BGCOLOR="{color}">
  <TR><TD COLSPAN="2"><B>{name}</B> ({tipo})</TD></TR>
  <TR><TD><FONT POINT-SIZE="10">Input</FONT></TD><TD><FONT POINT-SIZE="10">{input_shape}</FONT></TD></TR>
  <TR><TD><FONT POINT-SIZE="10">Output</FONT></TD><TD><FONT POINT-SIZE="10">{output_shape}</FONT></TD></TR>
</TABLE>>"""
            f.write(f'    {node_id} [label={label}];\n')

        for i in range(1, len(model.layers)):
            f.write(f'    layer_{i-1} -> layer_{i};\n')

        f.write("}\n")

# === Mostrar Visualización de la Arquitectura ===
def mostrar_arquitectura(modelo, dot_output_path, png_output_path):
    st.subheader("🕸️ Arquitectura Visual RNN")

    if not os.path.exists(png_output_path):
        try:
            generar_dot_con_tablas(modelo, dot_output_path)
            (graph,) = pydot.graph_from_dot_file(dot_output_path)
            graph.write_png(png_output_path)
        except Exception as e:
            st.warning(f"⚠️ No se pudo generar el diagrama: {e}")

    if os.path.exists(png_output_path):
        st.image(png_output_path, caption="Estructura de la red neuronal", use_container_width=True)

        with open(png_output_path, "rb") as file:
            st.download_button(
                label="💾 Guardar imagen del diagrama",
                data=file,
                file_name="modelo_arquitectura_23.png",
                mime="image/png"
            )

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
    modelo = cargar_modelo('modelo23.keras')
    mostrar_resumen_modelo(modelo)
    mostrar_arquitectura(modelo, "modelo_coloreado.dot", "modelo_coloreado.png")


# === Lanzar App ===
if __name__ == '__main__':
    main()
    mostrar_firma_sidebar()