"""VibraForma: protótipo de amplificação de movimento em vídeo.

O app usa fluxo óptico denso para estimar o deslocamento entre cada quadro e
o primeiro quadro, aplica um filtro temporal por FFT à faixa selecionada e
amplia esse deslocamento. O rastreamento de pontos fornece uma comparação em
pixels. Sem calibração, os valores não representam milímetros.
"""

from __future__ import annotations

import io
import hashlib
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="VibraForma | Inspeção visual", page_icon="〰️", layout="wide")


@st.cache_data(show_spinner=False, max_entries=3)
def decode_video(video_bytes: bytes, max_seconds: int, max_width: int):
    """Decodifica o começo do vídeo e reduz a resolução para caber no celular."""
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(video_bytes)
        source_path = temp.name
    capture = cv2.VideoCapture(source_path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        Path(source_path).unlink(missing_ok=True)
        raise ValueError("Não foi possível ler as dimensões do vídeo.")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Limita duração e quantidade de quadros para manter o FFT viável no protótipo.
    source_limit = max(2, int(max_seconds * fps))
    if total_frames > 0:
        source_limit = min(source_limit, total_frames)
    stride = max(1, int(np.ceil(source_limit / 180)))
    sample_limit = max(2, int(np.ceil(source_limit / stride)))
    effective_fps = fps / stride
    scale = min(1.0, max_width / width)
    new_size = (max(2, int(width * scale)), max(2, int(height * scale)))

    frames = []
    frame_index = 0
    while frame_index < source_limit and len(frames) < sample_limit:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            if new_size != (width, height):
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_index += 1
    capture.release()
    Path(source_path).unlink(missing_ok=True)
    return frames, effective_fps, stride, (width, height)


def temporal_bandpass(data: np.ndarray, fps: float, low_hz: float, high_hz: float):
    """Mantém apenas as componentes temporais dentro da faixa escolhida."""
    spectrum = np.fft.rfft(data, axis=0)
    frequencies = np.fft.rfftfreq(data.shape[0], d=1.0 / fps)
    keep = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(keep):
        raise ValueError("A faixa escolhida não contém frequências amostradas no vídeo.")
    spectrum[~keep] = 0
    return np.fft.irfft(spectrum, n=data.shape[0], axis=0).astype(np.float32)


def track_points(frames: list[np.ndarray], roi: tuple[int, int, int, int], max_points: int):
    """Rastreia características do ROI quadro a quadro com Lucas-Kanade."""
    x, y, w, h = roi
    first_gray = cv2.cvtColor(frames[0][y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    points = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=max_points,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )
    if points is None or len(points) == 0:
        return None, None

    points = points.astype(np.float32)
    initial = points[:, 0, :].copy()
    tracks = [initial.copy()]
    active_ids = np.arange(len(initial))
    previous_gray = first_gray
    current_points = points
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        if len(current_points) == 0:
            tracks.append(np.full_like(initial, np.nan))
            previous_gray = gray
            continue
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, current_points, None, **lk_params)
        if next_points is None or status is None:
            tracks.append(np.full_like(initial, np.nan))
            break
        good = status.reshape(-1).astype(bool) & np.isfinite(next_points[:, 0, :]).all(axis=1)
        tracked = np.full_like(initial, np.nan)
        tracked[active_ids[good]] = next_points[good, 0, :]
        tracks.append(tracked)
        # Pontos perdidos permanecem perdidos; evitar reinicialização mantém
        # comparável cada série ao mesmo ponto inicial.
        active_ids = active_ids[good]
        current_points = next_points[good]
        previous_gray = gray
    while len(tracks) < len(frames):
        tracks.append(np.full_like(initial, np.nan))
    return initial, np.stack(tracks, axis=0)


def magnify_video(
    frames: list[np.ndarray],
    roi: tuple[int, int, int, int],
    fps: float,
    low_hz: float,
    high_hz: float,
    gain: float,
    progress,
):
    """Amplifica o deslocamento filtrado do ROI e recompõe os quadros."""
    x, y, w, h = roi
    reference = cv2.cvtColor(frames[0][y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
    fields = []
    for index, frame in enumerate(frames):
        gray = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2GRAY)
        field = cv2.calcOpticalFlowFarneback(
            reference,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=21,
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        fields.append(field)
        progress((index + 1) / len(frames) * 0.55, text="Estimando deslocamentos no ROI…")

    flow = np.stack(fields, axis=0).astype(np.float32)
    filtered = temporal_bandpass(flow, fps, low_hz, high_hz)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    output_frames = []
    for index, frame in enumerate(frames):
        # O fluxo filtrado representa movimento adicional da faixa selecionada.
        # remap usa coordenadas de origem para cada pixel de destino.
        extra = max(0.0, gain - 1.0)
        map_x = xx - extra * filtered[index, :, :, 0]
        map_y = yy - extra * filtered[index, :, :, 1]
        source_roi = frame[y : y + h, x : x + w]
        amplified_roi = cv2.remap(
            source_roi,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        result = frame.copy()
        result[y : y + h, x : x + w] = amplified_roi
        output_frames.append(result)
        progress(0.55 + (index + 1) / len(frames) * 0.45, text="Gerando comparação…")
    return output_frames


def encode_mp4(frames: list[np.ndarray], fps: float) -> bytes:
    """Exporta MP4 H.264/yuv420p para reprodução no navegador."""
    height, width = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as folder:
        output_path = Path(folder) / "amplificado.mp4"
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
            "-r", str(fps), "-i", "pipe:0", "-an",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path),
        ]
        completed = subprocess.run(
            command, input=b"".join(frame.tobytes() for frame in frames),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError(f"Falha ao exportar o vídeo H.264: {detail}")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("A exportação não gerou um arquivo de vídeo válido.")
        return output_path.read_bytes()


st.title("VibraForma")
st.caption("Protótipo de amplificação visual e comparação por rastreamento de pontos")

st.info(
    "Use um vídeo com a câmera fixa e a estrutura em operação. O resultado é uma "
    "visualização para triagem: sem calibração e validação experimental, não é "
    "uma medição física nem um diagnóstico automático."
)

uploaded = st.file_uploader(
    "Importe um vídeo da estrutura em operação",
    type=["mp4", "mov", "avi", "m4v"],
    help="O protótipo processa até 12 segundos e reduz o vídeo para acelerar a análise.",
)

if uploaded:
    with st.spinner("Lendo o vídeo…"):
        try:
            video_bytes = uploaded.getvalue()
            video_id = hashlib.sha256(video_bytes).hexdigest()
            frames, fps, stride, original_size = decode_video(video_bytes, 12, 480)
        except Exception as error:
            st.error(f"Não foi possível abrir o vídeo: {error}")
            st.stop()

    if len(frames) < 8:
        st.error("O vídeo precisa ter pelo menos 8 quadros para esta análise.")
        st.stop()
    height, width = frames[0].shape[:2]
    nyquist = fps / 2.0
    st.caption(
        f"Vídeo original: {original_size[0]}×{original_size[1]} px · "
        f"análise: {width}×{height} px · {fps:.2f} quadros/s efetivos · "
        f"limite de Nyquist: {nyquist:.2f} Hz"
    )
    if nyquist < 0.5:
        st.error("A taxa de quadros é baixa demais para selecionar uma faixa útil de frequência.")
        st.stop()

    left, right = st.columns([1.05, 1])
    with left:
        st.subheader("1. Selecione a região")
        default_w = max(40, int(width * 0.6))
        default_h = max(40, int(height * 0.6))
        x = st.slider("Posição horizontal (x)", 0, max(0, width - 20), 0)
        y = st.slider("Posição vertical (y)", 0, max(0, height - 20), 0)
        w = st.slider("Largura da região", 20, width - x, min(default_w, width - x))
        h = st.slider("Altura da região", 20, height - y, min(default_h, height - y))
        roi = (x, y, w, h)
        preview = frames[0].copy()
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 220, 255), 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Região destacada em amarelo", use_container_width=True)

    with right:
        st.subheader("2. Configure a amplificação")
        highest = max(0.5, round(nyquist, 1))
        default_high = min(3.0, highest)
        default_low = min(2.0, max(0.1, default_high - 0.5))
        low_hz = st.number_input(
            "Frequência mínima (Hz)", min_value=0.1, max_value=highest,
            value=default_low, step=0.1
        )
        high_hz = st.number_input(
            "Frequência máxima (Hz)", min_value=0.1, max_value=highest,
            value=max(default_low + 0.1, default_high), step=0.1
        )
        gain = st.slider("Ganho de movimento", 1.0, 15.0, 5.0, 0.5)
        use_tracking = st.checkbox("Rastrear pontos e mostrar deslocamentos", value=True)
        max_points = st.slider("Quantidade máxima de pontos", 3, 30, 12, disabled=not use_tracking)
        st.caption("O ganho 1× não adiciona amplificação; 5× é um ponto de partida para inspeção visual.")

    if high_hz <= low_hz:
        st.warning("A frequência máxima precisa ser maior que a mínima.")
    elif high_hz > nyquist:
        st.warning(f"A frequência máxima não pode passar de {nyquist:.2f} Hz para este vídeo.")
    elif st.button("Processar vídeo", type="primary", use_container_width=True):
        progress = st.progress(0.0, text="Preparando…")
        try:
            amplified = magnify_video(frames, roi, fps, low_hz, high_hz, gain, progress.progress)
            output_bytes = encode_mp4(amplified, fps)
            initial_points = track_data = None
            if use_tracking:
                progress.progress(0.0, text="Rastreando pontos…")
                initial_points, track_data = track_points(frames, roi, max_points)
            progress.empty()
            st.session_state["result"] = {
                "video": output_bytes,
                "input_id": video_id,
                "initial_points": initial_points,
                "tracks": track_data,
                "fps": fps,
                "roi": roi,
                "low_hz": low_hz,
                "high_hz": high_hz,
                "gain": gain,
            }
        except Exception as error:
            progress.empty()
            st.error(f"Falha no processamento: {error}")

    result = st.session_state.get("result")
    if result and result.get("input_id") != video_id:
        result = None
    if result:
        st.divider()
        st.subheader("3. Compare o vídeo original e o amplificado")
        original_col, amplified_col = st.columns(2)
        with original_col:
            st.markdown("**Original**")
            st.video(video_bytes)
        with amplified_col:
            st.markdown(f"**Amplificado · {result['gain']:.1f}× · {result['low_hz']:.1f}–{result['high_hz']:.1f} Hz**")
            st.video(result["video"])
            st.download_button(
                "Baixar vídeo amplificado",
                data=result["video"],
                file_name="vibraforma_amplificado.mp4",
                mime="video/mp4",
            )

        if result["tracks"] is not None:
            st.subheader("4. Comparação pelo rastreamento de pontos")
            tracks = result["tracks"]
            starts = result["initial_points"]
            seconds = np.arange(len(tracks)) / result["fps"]
            series = {}
            for point_index in range(tracks.shape[1]):
                dx = tracks[:, point_index, 0] - starts[point_index, 0]
                dy = tracks[:, point_index, 1] - starts[point_index, 1]
                displacement = np.sqrt(dx**2 + dy**2)
                # Ignora pontos que o rastreador perdeu durante o vídeo.
                if np.isfinite(displacement).sum() >= max(5, len(displacement) // 3):
                    series[f"Ponto {point_index + 1}"] = displacement

            if series:
                chart = pd.DataFrame(series, index=np.round(seconds, 3))
                chart.index.name = "Tempo (s)"
                st.line_chart(chart, y_label="Deslocamento (pixels)")
                st.caption(
                    "O gráfico mostra o deslocamento aparente de cada ponto em relação à posição inicial. "
                    "Movimento da câmera também afeta a curva. Para converter para unidades físicas, "
                    "é necessário calibrar a escala da imagem e validar o método."
                )
                csv_buffer = io.StringIO()
                chart.to_csv(csv_buffer)
                st.download_button(
                    "Baixar dados dos pontos (CSV)",
                    data=csv_buffer.getvalue().encode("utf-8"),
                    file_name="vibraforma_rastreamento.csv",
                    mime="text/csv",
                )
            else:
                st.warning("Não foi possível manter pontos rastreáveis nesta região. Tente outra região com mais textura ou contraste.")

st.divider()
st.caption(
    "Método experimental para prova de conceito: fluxo óptico denso + filtro temporal FFT + "
    "rastreamento Lucas–Kanade. Resultados dependem da câmera, iluminação, textura e qualidade do vídeo."
)
