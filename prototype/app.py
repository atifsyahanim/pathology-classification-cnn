import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
from skimage.transform import resize

# === CONFIG ===
MEAN_CHANNEL = [-3.2865016, -0.0005206187, -0.00029896342]
STD_CHANNEL = [47.69045, 3.14551, 1.590327]
MODEL_PATH = "model_resnet18.h5"

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# === Feature Extraction ===
def extract_voice_features(audio, sr, hop_length=256):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40,
        n_fft=2048,
        hop_length=hop_length,
        n_mels=128,
        fmin=80,
        fmax=8000,
        htk=True
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.stack([mfcc, delta, delta2])

# === Normalize MFCC ===
def normalize_mfcc(mfcc_stack):
    norm_channels = []
    for i in range(3):
        norm = (mfcc_stack[i] - MEAN_CHANNEL[i]) / STD_CHANNEL[i]
        resized = resize(norm, (128, 128), mode='constant')
        norm_channels.append(resized)
    return np.stack(norm_channels, axis=-1)

# === Prediction Function ===
def predict_from_audio(audio_path):
    if audio_path is None:
        return "<div style='color: white;'>Please upload an audio file first.</div>"
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)

        max_amp = np.max(np.abs(y))
        if max_amp != 0:
            y = y / max_amp * 0.999

        mfcc_stack = extract_voice_features(y, sr)
        features = normalize_mfcc(mfcc_stack)
        input_tensor = np.expand_dims(features, axis=0)

        raw_score = model.predict(input_tensor)[0][0]
        label = "Abnormal" if raw_score > 0.5 else "Normal"
        confidence = raw_score if raw_score > 0.5 else 1 - raw_score

        if label == "Normal":
            status_message = "No abnormalities found in sample"
            color = "#2ecc71"
        else:
            status_message = ""
            color = "#e74c3c"

        result_html = f"""
        <div style="font-size:18px; line-height:1.6; color:white; text-align: left;">
            <div style="margin-bottom: 15px;">
                <b style="color: white;">Sample is </b>
                <span style="color:{color} !important; font-weight: bold; font-size: 20px;">{label}</span>
            </div>
            {f'<div style="color: {color}; font-style: italic; margin-bottom: 10px;">{status_message}</div>' if status_message else ''}
            <div style="font-size: 14px; color: #ccc;">
                <b>Confidence:</b> {confidence * 100:.1f}%<br>
                <b>Raw Score:</b> {raw_score:.2f}
            </div>
        </div>
        """
        return result_html
    except Exception as e:
        return f"<div style='color: white;'>Error processing audio: {str(e)}</div>"

# === Loading Function ===
def show_loading():
    return """
    <div style="text-align: center; color: white; font-size: 16px;">
        <div style="display: inline-block; animation: spin 1s linear infinite; font-size: 24px; margin-bottom: 10px;">
            ⚡
        </div>
        <br>
        <span>Wait while we're analyzing...</span>
        <style>
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    </div>
    """

# === Gradio Interface ===
custom_theme = gr.themes.Base(
    font=gr.themes.GoogleFont("Poppins"),
    primary_hue="blue",
    secondary_hue="yellow"
)

with gr.Blocks(theme=custom_theme, css="""
html, body, .gradio-container {
    background-color: #003366 !important;
    color: white !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
}

/* Remove scroll bars */
::-webkit-scrollbar {
    display: none;
}

html {
    -ms-overflow-style: none;
    scrollbar-width: none;
}

h1, h2, h3, p, .gr-markdown, span, div {
    color: white !important;
}

/* Fix all labels and text to be white */
label, .gr-label, .gr-text {
    color: white !important;
}

/* Audio component styling - Make ALL elements grey with clean design */
.gr-audio, .gr-audio *, 
.gr-file-preview, .gr-file-preview *,
.gr-dropzone, .gr-dropzone *,
.gr-upload, .gr-upload *,
button, button *,
.gr-button, .gr-button *,
input, textarea, select {
    color: #888 !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove container backgrounds */
.gr-audio {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Upload area clean styling */
.gr-dropzone {
    background: transparent !important;
    border: 2px dashed rgba(136, 136, 136, 0.3) !important;
    border-radius: 8px !important;
}

/* Upload button text and all interactive elements */
.gr-upload-button, .gr-upload-button *,
.gr-file-upload, .gr-file-upload *,
.gr-dropzone, .gr-dropzone *,
button[aria-label*="upload"], button[aria-label*="upload"] *,
button[type="button"], button[type="button"] * {
    color: #888 !important;
    background-color: transparent !important;
    border: none !important;
}

/* More aggressive label targeting */
.custom-audio label,
.custom-audio .gr-label,
.custom-audio span[data-testid="block-label"],
.gr-block .gr-form .gr-box label,
div[data-testid*="audio"] label,
div[data-testid*="audio"] .gr-label,
div[data-testid*="audio"] span,
div[data-testid*="audio"] p,
div[data-testid*="audio"] div,
div[data-testid*="audio"] button,
div[data-testid*="audio"] button * {
    color: #888 !important;
    background: transparent !important;
}

/* All possible audio element selectors */
.gr-audio .gr-upload-text, 
.gr-audio .gr-file-name,
.gr-audio .gr-file-size,
.gr-audio .gr-file-duration,
.gr-audio label,
.gr-audio .gr-label,
.gr-audio .gr-text,
.gr-audio .gr-button,
.gr-audio span,
.gr-audio div,
.gr-audio p,
.gr-audio button,
.gr-audio button * {
    color: #888 !important;
    background: transparent !important;
}

/* Speed display and duration styling */
.gr-audio .gr-audio-controls,
.gr-audio .gr-audio-controls *,
.gr-audio .gr-waveform,
.gr-audio .gr-waveform * {
    color: #888 !important;
    background: transparent !important;
}

/* Universal button and upload styling */
button, .gr-button {
    color: #888 !important;
    background-color: transparent !important;
    border: none !important;
}

/* Universal label override */
label {
    color: #888 !important;
    background: transparent !important;
}

/* Block label override */
.gr-block-label, .gr-block-label * {
    color: #888 !important;
    background: transparent !important;
}

/* Gradio form elements */
.gr-form label, .gr-form .gr-label, .gr-form span, .gr-form button, .gr-form button * {
    color: #888 !important;
    background: transparent !important;
}

/* Remove all container styling */
.gr-block, .gr-box, .gr-form {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Clean waveform container */
.gr-audio .gr-waveform-container {
    background: transparent !important;
    border: 1px solid rgba(136, 136, 136, 0.3) !important;
    border-radius: 4px !important;
}

#analyze-btn {
    background-color: #ffd500 !important;
    color: #003366 !important;
    font-weight: bold;
    font-size: 16px;
    padding: 10px 30px;
    border-radius: 8px;
    border: none;
    margin-top: 12px;
}

.gr-html-container {
    overflow: hidden !important;
}

/* Ensure all text is white */
* {
    color: inherit !important;
}

.gr-markdown * {
    color: white !important;
}
""") as app:

    # Title
    gr.Markdown("""
    <h1 style="text-align: center; font-size: 36px; margin-bottom: 5px; color: white !important;">AI-Based Voice Screening</h1>
    """)

    # Recommendation - Fixed to be white
    gr.Markdown("""
    <div style="text-align: center; font-size: 16px; color: white !important;">
        📁 <b style="color: white !important;">Recommended:</b> Use file length around 30 seconds (Must be WAV file)
    </div>
    """)

    # Audio upload - Fixed upload functionality
    with gr.Row():
        audio_input = gr.Audio(
            type="filepath",
            label="Upload WAV File",
            sources=["upload"]
        )

    # Analyze button
    with gr.Row():
        run_button = gr.Button("Analyze", elem_id="analyze-btn")

    # Result output
    with gr.Row():
        output_html = gr.HTML()

    # Disclaimer - Changed to grey color
    gr.Markdown("""
    <div style="font-size: 14px; padding-top: 25px; color: #aaa;">
        ⚠️ <i style="color: #aaa;">This AI tool only analyzes voice recordings and does not consider other clinical factors or symptoms. It is not a substitute for professional diagnosis. Please consult a qualified healthcare provider for any health concerns.</i>
    </div>
    """)

    run_button.click(
        fn=show_loading, 
        inputs=None, 
        outputs=output_html
    ).then(
        fn=predict_from_audio, 
        inputs=audio_input, 
        outputs=output_html
    )

app.launch(share=True)