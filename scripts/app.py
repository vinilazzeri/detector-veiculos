import streamlit as st
import tempfile
import os
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import time

# ------------------------------
# Controle de reset da aplicação
# ------------------------------
if "reset_app" not in st.session_state:
    st.session_state.reset_app = False

if st.session_state.reset_app:
    # Limpa todo o cache
    st.cache_resource.clear()
    st.cache_data.clear()
    
    # Limpa a session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    st.experimental_rerun()

# Configuração da página
st.set_page_config(
    page_title="Detector de Veículos", 
    page_icon="♦️", 
    layout="wide"
)

# Solução para possíveis erros do PyTorch/Streamlit
try:
    import torch
    torch.utils.import_ir_module = lambda *args, **kwargs: None
except ImportError:
    pass

# Carrega o modelo YOLO
@st.cache_resource
def load_model():
    try:
        model = YOLO("runs/detect/train/weights/best.pt")
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        return None

model = load_model()

# Funções de processamento
def process_image(image):
    try:
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            original_image = np.array(image)

        results = model(original_image)
        result_image = results[0].plot()
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_image_rgb)
    except Exception as e:
        st.error(f"Erro ao processar imagem: {str(e)}")
        return None

def process_video(video_path):
    try:
        # Verifica se o vídeo é válido
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Não foi possível abrir o vídeo")
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            cap.release()
            raise Exception("Vídeo não contém frames ou está corrompido")
            
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Processa o vídeo
        cap = cv2.VideoCapture(video_path)
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            plotted_frame = results[0].plot()
            out.write(plotted_frame)

            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            progress_bar.progress(min(progress, 100))
            status_text.text(f"Processando vídeo... {min(progress, 100)}% concluído")

        cap.release()
        out.release()

        # Verifica se o vídeo processado é válido
        cap = cv2.VideoCapture(output_path)
        if not cap.isOpened() or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
            cap.release()
            os.unlink(output_path)
            raise Exception("Vídeo processado está vazio ou corrompido")
        cap.release()

        return output_path
    except Exception as e:
        st.error(f"Erro durante o processamento do vídeo: {str(e)}")
        if 'output_path' in locals() and output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        return None
    finally:
        if 'cap' in locals():
            cap.release()
        if 'out' in locals():
            out.release()

# Interface
try:
    display_logo = "../imgs/logo.png"
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        # Carrega a imagem com PIL para controle de qualidade
        logo = Image.open(display_logo)
        st.image(logo, width=800, output_format='PNG')

    input_type = st.radio("Selecione o tipo de entrada:", 
                         ["Imagem", "Vídeo"],
                         horizontal=True)

    uploaded_file = st.file_uploader(
        f"Carregue um arquivo de {input_type.lower()}",
        type=["jpg", "jpeg", "png"] if input_type == "Imagem" else ["mp4", "mov"]
    )

    if uploaded_file is not None:
        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 6, 1.5])
        with btn_col2:
            if st.button("Gerar Detecção", type="secondary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    if input_type == "Imagem":
                        image = Image.open(uploaded_file)

                        progress_bar.progress(30)
                        status_text.text("Processando imagem...")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Imagem Original")
                            st.image(image, use_container_width=True, caption="Imagem carregada")

                        progress_bar.progress(60)
                        result = process_image(image)
                        progress_bar.progress(90)

                        if result is None:
                            raise Exception("Falha no processamento da imagem")

                        with col2:
                            st.subheader("Resultado da Detecção")
                            st.image(result, use_container_width=True, caption="Detecções YOLOv8")

                        progress_bar.progress(100)

                        st.markdown("---")
                        dl_col1, dl_col2, dl_col3 = st.columns([1.5, 6, 1.5])
                        with dl_col2:
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                                result.save(tmp.name, quality=95)  
                                with open(tmp.name, "rb") as file:
                                    st.download_button(
                                        label="⬇️ Baixar Imagem Processada",
                                        data=file,
                                        file_name="detection_result.jpg",
                                        mime="image/jpeg",
                                        use_container_width=True
                                    )

                            if st.button("🔄 Recarregar Página", type="secondary", use_container_width=True, key="reload_img"):
                                st.session_state.reset_app = True
                                st.experimental_rerun()

                    elif input_type == "Vídeo":
                        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                            tmp.write(uploaded_file.read())
                            video_path = tmp.name

                        # Verifica se o vídeo é válido
                        cap_test = cv2.VideoCapture(video_path)
                        if not cap_test.isOpened():
                            cap_test.release()
                            raise Exception("Vídeo inválido - não pode ser aberto")
                            
                        frame_count = int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT))
                        if frame_count == 0:
                            cap_test.release()
                            raise Exception("Vídeo não contém frames")
                        cap_test.release()

                        st.subheader("Vídeo Original")
                        st.video(video_path)

                        output_path = process_video(video_path)

                        if output_path and os.path.exists(output_path):
                            # Verifica se o vídeo processado é válido
                            cap_test = cv2.VideoCapture(output_path)
                            if not cap_test.isOpened() or int(cap_test.get(cv2.CAP_PROP_FRAME_COUNT)) == 0:
                                cap_test.release()
                                raise Exception("Vídeo processado está vazio ou corrompido")
                            cap_test.release()

                            success_message = st.empty()
                            success_message.success("✅ Processamento concluído com sucesso!")
                            
                            time.sleep(2)
                            success_message.empty()

                            with open(output_path, 'rb') as f:
                                video_bytes = f.read()

                            st.markdown("---")
                            dl_col1, dl_col2, dl_col3 = st.columns([1.5, 6, 1.5])
                            with dl_col2:
                                st.download_button(
                                    label="Baixar Vídeo Processado",
                                    data=video_bytes,
                                    file_name="detection_result.mp4",
                                    mime="video/mp4",
                                    use_container_width=True,
                                    type="primary"
                                )

                                if st.button("🔄 Recarregar Aplicação", type="secondary", use_container_width=True, key="reload_video"):
                                    st.session_state.reset_app = True
                                    st.experimental_rerun()
                        else:
                            raise Exception("Falha no processamento do vídeo")

                except Exception as e:
                    st.error(f"Erro durante o processamento: {str(e)}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
                    # Limpeza de arquivos temporários
                    if input_type == "Vídeo" and 'video_path' in locals() and video_path and os.path.exists(video_path):
                        try:
                            os.unlink(video_path)
                        except:
                            pass
                    if 'output_path' in locals() and output_path and os.path.exists(output_path):
                        try:
                            os.unlink(output_path)
                        except:
                            pass
                    if 'cap_test' in locals():
                        cap_test.release()

except Exception as e:
    st.error(f"Erro inesperado na aplicação: {str(e)}")