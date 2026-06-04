import os
import json
import logging
import random
import tempfile
import asyncio
import re
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import httpx
import requests
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import matplotlib.pyplot as plt
from json_repair import repair_json  # pip install json-repair

# Suppress oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# Setup logging ke terminal
logging.basicConfig(level=logging.INFO)

# Konfigurasi Gemini API (ganti dengan key asli Anda jika perlu)
GEMINI_API_KEY = "...dkKNe7.....xqEBVs..."  # ← ganti dengan API key kamu
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# ── Download model dari Google Drive jika belum ada ──
MODEL_PATH = "skin_model.h5"
GDRIVE_FILE_ID = "1Tm1OpVsGDvGtCHNaq4xu_DVNzjCSZO3X"

def download_model_from_gdrive(file_id: str, dest_path: str):
    """Download file dari Google Drive — hanya dipakai sebagai fallback lokal.
    Di production (Railway), model sudah di-bake ke Docker image saat build time."""
    import gdown
    logging.info(f"Mengunduh model dari Google Drive ke '{dest_path}'...")
    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url, dest_path, quiet=False, fuzzy=True, use_cookies=False)
    if result is None:
        raise RuntimeError(
            "gdown gagal. Model seharusnya sudah ada di image (di-download saat docker build)."
        )
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    logging.info(f"Model berhasil diunduh ke '{dest_path}' ({size_mb:.1f} MB)")

def verify_and_load_model(model_path: str, file_id: str, max_retries: int = 3):
    """Verifikasi file HDF5 valid, re-download jika corrupt, lalu load model.
    max_retries mencegah loop tak terbatas jika Google Drive terus mengembalikan HTML."""
    import h5py

    for attempt in range(max_retries):
        if not os.path.exists(model_path):
            download_model_from_gdrive(file_id, model_path)

        valid = False
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            if size_mb > 10:
                try:
                    with h5py.File(model_path, "r") as f:
                        valid = True
                        logging.info(f"File HDF5 valid ({size_mb:.1f} MB)")
                except Exception as e:
                    logging.error(f"File corrupt/bukan HDF5: {e}")
            else:
                logging.error(f"File terlalu kecil ({size_mb:.1f} MB) — kemungkinan HTML bukan model")

        if valid:
            break

        logging.warning(f"Percobaan {attempt + 1}/{max_retries}: menghapus file corrupt dan mengunduh ulang...")
        if os.path.exists(model_path):
            os.remove(model_path)

        if attempt == max_retries - 1:
            raise RuntimeError(
                f"Model gagal dimuat setelah {max_retries} percobaan. "
                "Kemungkinan: (1) GDrive quota habis, (2) file tidak publik, (3) ID salah. "
                "Solusi: gunakan Railway Volume atau Hugging Face Hub."
            )

    logging.info("Loading model...")
    return tf.keras.models.load_model(model_path)

# Load model klasifikasi kulit dan label
model = verify_and_load_model(MODEL_PATH, GDRIVE_FILE_ID)
labels = ["Acne", "Dry", "Normal", "Oily"]

# Simpan gambar terakhir yang dianalisis (key: session_id atau "latest")
latest_frame_store: dict = {}  # {"image_bytes": bytes, "box": [...], "box_color": str}

# Load face detector dari MediaPipe
face_detector = mp.tasks.vision.FaceDetector.create_from_options(
    mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="face_detection_short_range.tflite"),
        min_detection_confidence=0.75
    )
)

# ---------- Helper functions ----------
def get_static_recommendation(skin_type):
    """Rekomendasi statis (fallback) jika AI tidak tersedia."""
    if skin_type == "Acne":
        return {
            "rec": "• Salicylic Acid 2%\n• Niacinamide Serum\n• Gentle Cleanser\n• Oil-Free Moisturizer\n• Sunscreen SPF 50",
            "routine": "☀️ Pagi: Gentle cleanser → Niacinamide → Moisturizer → SPF\n🌙 Malam: Cleanse → Salicylic acid → Moisturizer"
        }
    if skin_type == "Dry":
        return {
            "rec": "• Hyaluronic Acid\n• Ceramide Cream\n• Hydrating Cleanser\n• Night Cream",
            "routine": "☀️ Pagi: Hydrating cleanser → Hyaluronic acid → Moisturizer → SPF\n🌙 Malam: Gentle cleanser → Hydrating serum → Night cream"
        }
    if skin_type == "Oily":
        return {
            "rec": "• Oil-control Cleanser\n• Niacinamide\n• Gel Moisturizer\n• Non-comedogenic SPF",
            "routine": "☀️ Pagi: Oil control wash → Niacinamide → Gel moisturizer → SPF\n🌙 Malam: Face wash → Salicylic serum → Light moisturizer"
        }
    return {
        "rec": "• Maintain routine\n• Daily SPF\n• Hydration\n• Light moisturizer",
        "routine": "☀️ Pagi: Gentle cleanse → Moisturizer → SPF\n🌙 Malam: Cleanse → Night cream"
    }

def process_frame(frame_bytes):
    """Deteksi wajah, klasifikasi jenis kulit, dan hitung parameter kulit."""
    np_arr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_detector.detect(mp_image)

    h, w, _ = frame.shape
    detection_data = None

    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)

            margin_x = int(bw * 0.10)
            margin_y = int(bh * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + bw + margin_x)
            y2 = min(h, y + bh + margin_y)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocessing untuk model klasifikasi
            img = cv2.resize(face, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img, verbose=0)
            class_id = np.argmax(pred)
            confidence = int(np.max(pred) * 100)
            skin_type = labels[class_id]

            # Simulasi skor parameter kulit (untuk demo)
            moisture = random.randint(60, 95)
            oil = random.randint(35, 90)
            acne = random.randint(5, 80)
            blackspot = random.randint(5, 70)
            wrinkle = random.randint(5, 60)

            now = datetime.now()
            current_date = now.strftime("%d-%m-%Y")
            current_time = now.strftime("%H:%M:%S")

            # Warna bounding box berdasarkan jenis kulit
            if skin_type == "Acne":
                box_color = "#FF0000"
            elif skin_type == "Oily":
                box_color = "#FFFF00"
            elif skin_type == "Dry":
                box_color = "#FF7800"
            else:
                box_color = "#00FF00"

            detection_data = {
                "skin_type": skin_type,
                "confidence": confidence,
                "moisture": moisture,
                "oil": oil,
                "acne": acne,
                "blackspot": blackspot,
                "wrinkle": wrinkle,
                "date": current_date,
                "time": current_time,
                "box": [x1, y1, x2, y2],
                "box_color": box_color
            }
            # Simpan frame + box untuk PDF
            latest_frame_store["image_bytes"] = frame_bytes
            latest_frame_store["box"] = [x1, y1, x2, y2]
            latest_frame_store["box_color"] = box_color
            latest_frame_store["skin_type"] = skin_type
            latest_frame_store["confidence"] = confidence
            break

    return detection_data, frame.shape[:2] if detection_data else None

# ---------- API Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    html_content = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=yes">
    <title>Facial Skin Detection | Smart Skincare Analysis</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Poppins', system-ui; background: linear-gradient(145deg, #fdf2f8 0%, #fbe9f2 100%); transition: background 0.3s; min-height: 100vh; }
        body.dark { background: linear-gradient(145deg, #121212 0%, #1e1e2a 100%); }
        .dashboard { display: flex; gap: 1.25rem; padding: 1.25rem; max-width: 1600px; margin: 0 auto; }
        .sidebar { width: 300px; background: rgba(255,255,255,0.85); backdrop-filter: blur(12px); border-radius: 2rem; padding: 1.5rem; display: flex; flex-direction: column; gap: 1.25rem; position: sticky; top: 1.25rem; }
        body.dark .sidebar { background: rgba(30,30,40,0.85); }
        .logo { font-size: 1.7rem; font-weight: 800; background: linear-gradient(135deg, #F45D9C, #FF9F3D); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .section-title { font-size: 1rem; font-weight: 600; color: #F45D9C; text-transform: uppercase; margin: 0.75rem 0 0.25rem 0; }
        .user-input { background: rgba(255,247,251,0.9); border: 1.5px solid rgba(244,93,156,0.3); border-radius: 1.5rem; padding: 0.75rem 1rem; width: 100%; margin-top: 0.5rem; }
        .buttons { display: flex; flex-direction: column; gap: 0.7rem; }
        .sidebar-btn { background: rgba(255,255,255,0.7); border: none; border-radius: 2rem; padding: 0.85rem 1rem; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 0.7rem; }
        .sidebar-btn:hover { background: #F45D9C; color: white; }
        .sidebar-btn.active { background: #F45D9C; color: white; }
        .camera-status { background: rgba(244,93,156,0.1); border-radius: 2rem; text-align: center; padding: 0.75rem; color: #F45D9C; margin-top: auto; }
        .center { flex: 1; display: flex; flex-direction: column; gap: 1.25rem; }
        .header { background: rgba(255,255,255,0.75); backdrop-filter: blur(10px); border-radius: 2rem; text-align: center; padding: 1.2rem; font-weight: 700; color: #F45D9C; }
        .video-container { background: rgba(255,255,255,0.5); border-radius: 2rem; position: relative; overflow: hidden; width: 100%; aspect-ratio: 4/3; }
        #canvasPreview { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; background: #f0eef3; display: block; }
        #video { display: none; }
        .placeholder { font-size: 1.2rem; color: #aaa; text-align: center; z-index: 2; }
        .right-panel { width: 340px; background: rgba(255,255,255,0.75); backdrop-filter: blur(12px); border-radius: 2rem; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; position: sticky; top: 1.25rem; }
        .analysis-title { font-size: 1.5rem; font-weight: 800; color: #F45D9C; }
        .confidence { font-size: 3.2rem; font-weight: 800; text-align: center; background: linear-gradient(135deg, #F45D9C, #ffad7a); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .skin-condition { font-size: 1.4rem; font-weight: 700; text-align: center; background: #F45D9C20; display: inline-block; margin: 0 auto; padding: 0.3rem 1rem; border-radius: 3rem; }
        .datetime { display: flex; justify-content: space-between; font-size: 0.75rem; background: rgba(0,0,0,0.03); padding: 0.5rem; border-radius: 1rem; }
        .progress-item { margin: 0.2rem 0; }
        .progress-item label { font-weight: 600; font-size: 0.8rem; display: flex; justify-content: space-between; }
        progress { width: 100%; height: 8px; border-radius: 20px; overflow: hidden; margin-top: 5px; }
        progress::-webkit-progress-bar { background: #e9e9ef; }
        progress::-webkit-progress-value { background: #F45D9C; }
        .rec-box { background: rgba(244,93,156,0.08); border-radius: 1.5rem; padding: 1rem; font-size: 0.75rem; line-height: 1.5; white-space: pre-line; }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.6); align-items: center; justify-content: center; }
        .modal-content { background: white; padding: 1.5rem; border-radius: 2rem; width: 90%; max-width: 550px; }
        body.dark .modal-content { background: #1f1f2e; color: #eee; }
        .close, .close-chart { float: right; font-size: 1.8rem; cursor: pointer; }
        @media (max-width: 1100px) { .sidebar { width: 100%; flex-direction: row; flex-wrap: wrap; } .right-panel { width: 100%; } }

        /* ===== DARK MODE: semua teks berubah putih/terang ===== */
        body.dark { color: #f0f0f0; }
        body.dark .sidebar { color: #f0f0f0; }
        body.dark .right-panel { color: #f0f0f0; background: rgba(30,30,40,0.85); }
        body.dark .section-title { color: #ff85bc; }
        body.dark .header { color: #ff85bc; background: rgba(30,30,40,0.75); }
        body.dark .analysis-title { color: #ff85bc; }
        body.dark .skin-condition { color: #f0f0f0; background: rgba(244,93,156,0.25); }
        body.dark .datetime { color: #c8c8d8; background: rgba(255,255,255,0.06); }
        body.dark .datetime span { color: #c8c8d8; }
        body.dark .progress-item label { color: #e0e0f0; }
        body.dark .progress-item label span { color: #e0e0f0; }
        body.dark progress::-webkit-progress-bar { background: #333350; }
        body.dark progress::-webkit-progress-value { background: linear-gradient(90deg, #F45D9C, #ff9f3d); }
        body.dark .rec-box { background: rgba(244,93,156,0.12); color: #eee; border: 1px solid rgba(244,93,156,0.2); }
        body.dark .user-input { background: rgba(30,30,50,0.9); border-color: rgba(244,93,156,0.4); color: #f0f0f0; }
        body.dark .user-input::placeholder { color: #888; }
        body.dark .sidebar-btn { background: rgba(40,40,60,0.8); color: #e0e0f0; }
        body.dark .sidebar-btn:hover { background: #F45D9C; color: white; }
        body.dark .sidebar-btn.active { background: #F45D9C; color: white; }
        body.dark .camera-status { color: #ff85bc; background: rgba(244,93,156,0.15); }
        body.dark .detect-info { color: #e0e0f0; }
        body.dark .placeholder { color: #888; }
        body.dark .modal-content h3 { color: #ff85bc; }
        body.dark .modal-content pre { color: #e0e0f0; }
        body.dark .close, body.dark .close-chart { color: #e0e0f0; }
    </style>
</head>
<body>
<div class="dashboard">
    <aside class="sidebar">
        <div class="logo">✨ Facial Skin Detection</div>
        <div class="user-section">
            <div class="section-title">👤 User</div>
            <input type="text" id="nameInput" placeholder="Nama lengkap" class="user-input">
            <input type="text" id="ageInput" placeholder="Umur" class="user-input">
        </div>
        <div class="buttons">
            <button id="btnOpen" class="sidebar-btn active">📷 Open Camera</button>
            <button id="btnPause" class="sidebar-btn">⏸ Pause</button>
            <button id="btnClose" class="sidebar-btn">❌ Close Camera</button>
            <button id="btnUpload" class="sidebar-btn">☁ Upload Image</button>
            <button id="btnClear" class="sidebar-btn">🗑 Clear Data</button>
            <button id="btnHistory" class="sidebar-btn">🕒 History</button>
            <button id="btnResult" class="sidebar-btn">📊 Analysis Chart</button>
            <button id="btnPdf" class="sidebar-btn">📄 Download PDF</button>
            <button id="btnDark" class="sidebar-btn">🌙 Dark Mode</button>
        </div>
        <div class="camera-status" id="cameraStatus">🔴 Offline</div>
    </aside>
    <main class="center">
        <div class="header">🌟 AI-Powered Skin Analysis 🌟</div>
        <div class="video-container">
            <video id="video" autoplay playsinline muted style="display: none;"></video>
            <canvas id="canvasPreview" width="860" height="620"></canvas>
            <div class="placeholder" id="placeholder">📸 Camera Preview</div>
        </div>
        <div class="detect-info" id="detectInfo">✨ Ready to analyze your skin</div>
    </main>
    <aside class="right-panel">
        <div class="analysis-title">📊 Live Results</div>
        <div class="confidence" id="confidenceScore">0%</div>
        <div class="skin-condition" id="skinCondition">—</div>
        <div class="datetime"><span>📅 <span id="dateVal">-</span></span><span>⏰ <span id="timeVal">-</span></span></div>
        <div class="progress-item"><label>💧 Moisture <span id="moistureVal">0%</span></label><progress id="moistureBar" value="0" max="100"></progress></div>
        <div class="progress-item"><label>🧴 Sebum <span id="oilVal">0%</span></label><progress id="oilBar" value="0" max="100"></progress></div>
        <div class="progress-item"><label>🔴 Acne <span id="acneVal">0%</span></label><progress id="acneBar" value="0" max="100"></progress></div>
        <div class="progress-item"><label>⚫ Blackspot <span id="blackspotVal">0%</span></label><progress id="blackspotBar" value="0" max="100"></progress></div>
        <div class="progress-item"><label>〰️ Wrinkle <span id="wrinkleVal">0%</span></label><progress id="wrinkleBar" value="0" max="100"></progress></div>
        <div><div class="section-title">💡 Smart Recommendation</div><div class="rec-box" id="recommendationText">Belum ada rekomendasi</div></div>
        <div><div class="section-title">🧼 Daily Routine</div><div class="rec-box" id="routineText">—</div></div>
    </aside>
</div>
<div id="historyModal" class="modal"><div class="modal-content"><span class="close">&times;</span><h3>📜 History Analysis</h3><pre id="historyList"></pre></div></div>
<div id="chartModal" class="modal"><div class="modal-content"><span class="close-chart">&times;</span><canvas id="resultChart" width="500" height="300"></canvas></div></div>
<script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvasPreview');
    const ctx = canvas.getContext('2d');
    const placeholder = document.getElementById('placeholder');
    const nameInput = document.getElementById('nameInput');
    const ageInput = document.getElementById('ageInput');
    const cameraStatus = document.getElementById('cameraStatus');
    const detectInfo = document.getElementById('detectInfo');
    const confidenceSpan = document.getElementById('confidenceScore');
    const skinConditionSpan = document.getElementById('skinCondition');
    const dateSpan = document.getElementById('dateVal');
    const timeSpan = document.getElementById('timeVal');
    const moistureBar = document.getElementById('moistureBar');
    const oilBar = document.getElementById('oilBar');
    const acneBar = document.getElementById('acneBar');
    const blackspotBar = document.getElementById('blackspotBar');
    const wrinkleBar = document.getElementById('wrinkleBar');
    const moistureVal = document.getElementById('moistureVal');
    const oilVal = document.getElementById('oilVal');
    const acneVal = document.getElementById('acneVal');
    const blackspotVal = document.getElementById('blackspotVal');
    const wrinkleVal = document.getElementById('wrinkleVal');
    const recommendationDiv = document.getElementById('recommendationText');
    const routineDiv = document.getElementById('routineText');

    const btnOpen = document.getElementById('btnOpen');
    const btnPause = document.getElementById('btnPause');
    const btnClose = document.getElementById('btnClose');
    const btnUpload = document.getElementById('btnUpload');
    const btnClear = document.getElementById('btnClear');
    const btnHistory = document.getElementById('btnHistory');
    const btnResult = document.getElementById('btnResult');
    const btnPdf = document.getElementById('btnPdf');
    const btnDark = document.getElementById('btnDark');

    let stream = null, ws = null, cameraActive = false, paused = false, animationId = null, currentResult = null, historyList = [];
    let lastSendTime = 0;
    const SEND_INTERVAL = 100; // kirim lebih sering = deteksi lebih responsif

    // Box tracking — pisah antara target (dari server) dan displayed (smooth)
    let lastBox = null;       // metadata warna/label
    let targetBox = null;     // koordinat terakhir dari server (piksel canvas)
    let smoothBox = null;     // koordinat yang ditampilkan (di-lerp tiap frame)
    let offscreenCanvas = null; // canvas khusus untuk kirim ke server (tanpa box overlay)

    function getStaticRecommendation(skinType) {
        if (skinType === "Acne") return { rec: "• Salicylic Acid 2%\\n• Niacinamide Serum\\n• Gentle Cleanser\\n• Oil-Free Moisturizer\\n• Sunscreen SPF 50", routine: "☀️ Pagi: Gentle cleanser → Niacinamide → Moisturizer → SPF\\n🌙 Malam: Cleanse → Salicylic acid → Moisturizer" };
        if (skinType === "Dry") return { rec: "• Hyaluronic Acid\\n• Ceramide Cream\\n• Hydrating Cleanser\\n• Night Cream", routine: "☀️ Pagi: Hydrating cleanser → Hyaluronic acid → Moisturizer → SPF\\n🌙 Malam: Gentle cleanser → Hydrating serum → Night cream" };
        if (skinType === "Oily") return { rec: "• Oil-control Cleanser\\n• Niacinamide\\n• Gel Moisturizer\\n• Non-comedogenic SPF", routine: "☀️ Pagi: Oil control wash → Niacinamide → Gel moisturizer → SPF\\n🌙 Malam: Face wash → Salicylic serum → Light moisturizer" };
        return { rec: "• Maintain routine\\n• Daily SPF\\n• Hydration\\n• Light moisturizer", routine: "☀️ Pagi: Gentle cleanse → Moisturizer → SPF\\n🌙 Malam: Cleanse → Night cream" };
    }

    let lastAICallTime = 0;
    let lastAISkinType = null;
    const AI_COOLDOWN_MS = 8000; // 8 detik cooldown agar tidak spam API

    async function upgradeRecommendationWithAI(result) {
        if (!result) return;
        const now = Date.now();
        // Lewati jika masih dalam cooldown DAN skin_type sama
        if (now - lastAICallTime < AI_COOLDOWN_MS && lastAISkinType === result.skin_type) return;
        lastAICallTime = now;
        lastAISkinType = result.skin_type;

        recommendationDiv.innerText = "⏳ Memuat rekomendasi AI...";
        routineDiv.innerText = "⏳ Memuat rutinitas AI...";
        try {
            const response = await fetch('/recommendations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    skin_type: result.skin_type,
                    confidence: result.confidence,
                    moisture: result.moisture,
                    oil: result.oil,
                    acne: result.acne,
                    blackspot: result.blackspot,
                    wrinkle: result.wrinkle,
                    name: nameInput.value || "User",
                    age: ageInput.value || "tidak diketahui"
                })
            });
            if (!response.ok) throw new Error('Gagal mengambil rekomendasi');
            const aiRec = await response.json();
            recommendationDiv.innerText = aiRec.rec || getStaticRecommendation(result.skin_type).rec;
            routineDiv.innerText = aiRec.routine || getStaticRecommendation(result.skin_type).routine;
        } catch (err) {
            console.warn("Rekomendasi AI tidak tersedia, menggunakan statis:", err.message);
            const fallback = getStaticRecommendation(result.skin_type);
            recommendationDiv.innerText = fallback.rec;
            routineDiv.innerText = fallback.routine;
        }
    }

    function updateUI(result) {
        if (!result) return;
        currentResult = result;
        confidenceSpan.innerText = `${result.confidence}%`;
        skinConditionSpan.innerText = result.skin_type;
        dateSpan.innerText = result.date;
        timeSpan.innerText = result.time;
        moistureBar.value = result.moisture; moistureVal.innerText = result.moisture+"%";
        oilBar.value = result.oil; oilVal.innerText = result.oil+"%";
        acneBar.value = result.acne; acneVal.innerText = result.acne+"%";
        blackspotBar.value = result.blackspot; blackspotVal.innerText = result.blackspot+"%";
        wrinkleBar.value = result.wrinkle; wrinkleVal.innerText = result.wrinkle+"%";
        // Tampilkan rekomendasi statis dulu sebagai fallback instan
        const { rec, routine } = getStaticRecommendation(result.skin_type);
        // Hanya ganti teks jika belum ada hasil AI atau skin_type berubah
        if (lastAISkinType !== result.skin_type) {
            recommendationDiv.innerText = rec;
            routineDiv.innerText = routine;
        }
        // Panggil AI hanya jika cooldown sudah habis
        const now = Date.now();
        if (now - lastAICallTime >= AI_COOLDOWN_MS || lastAISkinType !== result.skin_type) {
            upgradeRecommendationWithAI(result);
        }
        const entry = `${result.date} | ${result.time} | ${result.skin_type} | ${result.confidence}%`;
        if (historyList.length===0 || historyList[historyList.length-1]!==entry) { historyList.push(entry); if(historyList.length>50) historyList.shift(); }
    }

    // Dipakai untuk mode upload foto (static, bukan live)
    function drawBoundingBox(box, color, label, imageWidth, imageHeight) {
        const srcW = imageWidth  || canvas.width;
        const srcH = imageHeight || canvas.height;
        const scaleX = canvas.width  / srcW;
        const scaleY = canvas.height / srcH;
        const [x1,y1,x2,y2] = box;
        const sx1=x1*scaleX, sy1=y1*scaleY, sw=(x2-x1)*scaleX, sh=(y2-y1)*scaleY;
        ctx.strokeStyle = color;
        ctx.lineWidth   = 3;
        ctx.strokeRect(sx1, sy1, sw, sh);
        ctx.font = "bold 15px 'Poppins', sans-serif";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = color + 'cc';
        ctx.fillRect(sx1 - 1, sy1 - 22, textW + 10, 22);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, sx1 + 4, sy1 - 5);
    }


    function handleWebSocketMessage(event) {
        const res = JSON.parse(event.data);
        if (res.error) { detectInfo.innerText = "⚠️ No face detected"; return; }
        detectInfo.innerText = "✅ Skin analyzed!";
        updateUI(res);
        // Konversi koordinat box ke piksel canvas sekarang, simpan sebagai target
        if (res.box) {
            const [x1,y1,x2,y2] = res.box;
            // box dari server sudah dalam ruang canvas (offscreen = resolusi sama)
            targetBox = { x1, y1, x2, y2 };
            lastBox = { color: res.box_color, label: `${res.skin_type} (${res.confidence}%)` };
            // Inisialisasi smoothBox ke target jika belum ada (frame pertama)
            if (!smoothBox) smoothBox = { ...targetBox };
        } else {
            // Wajah tidak terdeteksi — fade out box
            targetBox = null;
        }
    }

    // Lerp helper — gerakkan nilai a menuju b dengan faktor t
    function lerp(a, b, t) { return a + (b - a) * t; }

    // Render loop: 60fps — gambar video bersih lalu overlay box yang di-smooth
    function renderLoop() {
        if (!cameraActive) return;
        if (video.videoWidth) {
            // Gambar frame video bersih (tanpa box) ke canvas tampil
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Smooth box: gerakkan smoothBox mendekati targetBox tiap frame
            if (targetBox && lastBox) {
                const t = 0.35; // faktor lerp: 0=tidak gerak, 1=langsung snap
                if (!smoothBox) smoothBox = { ...targetBox };
                smoothBox.x1 = lerp(smoothBox.x1, targetBox.x1, t);
                smoothBox.y1 = lerp(smoothBox.y1, targetBox.y1, t);
                smoothBox.x2 = lerp(smoothBox.x2, targetBox.x2, t);
                smoothBox.y2 = lerp(smoothBox.y2, targetBox.y2, t);
                drawSmoothBox(smoothBox, lastBox.color, lastBox.label);
            } else if (!targetBox) {
                // Wajah hilang — reset smooth box
                smoothBox = null;
            }
        }
        animationId = requestAnimationFrame(renderLoop);
    }

    // Gambar box dari koordinat smooth (sudah dalam piksel canvas, scale 1:1)
    function drawSmoothBox(sb, color, label) {
        const x = sb.x1, y = sb.y1, w = sb.x2 - sb.x1, h = sb.y2 - sb.y1;
        ctx.strokeStyle = color;
        ctx.lineWidth   = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.font = "bold 15px 'Poppins', sans-serif";
        const textW = ctx.measureText(label).width;
        ctx.fillStyle = color + 'cc';
        ctx.fillRect(x - 1, y - 22, textW + 10, 22);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 4, y - 5);
    }

    // Send loop: kirim frame BERSIH (dari video langsung) ke server
    // Menggunakan offscreenCanvas — server tidak melihat box overlay
    async function sendFrameToServer() {
        if (!cameraActive || !stream) return;
        if (!paused && ws && ws.readyState === WebSocket.OPEN && video.videoWidth) {
            const now = Date.now();
            if (now - lastSendTime >= SEND_INTERVAL) {
                lastSendTime = now;
                // Gambar video langsung ke offscreen canvas — tanpa box overlay
                offscreenCanvas.width  = canvas.width;
                offscreenCanvas.height = canvas.height;
                offscreenCanvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                const blob = await new Promise(resolve => offscreenCanvas.toBlob(resolve, 'image/jpeg', 0.8));
                if (ws && ws.readyState === WebSocket.OPEN) ws.send(blob);
            }
        }
        setTimeout(sendFrameToServer, SEND_INTERVAL);
    }

    async function startCamera() {
        if (cameraActive) return;
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            await video.play();

            // Sinkronkan canvas ke resolusi asli video agar koordinat box 1:1
            canvas.width  = video.videoWidth  || 860;
            canvas.height = video.videoHeight || 620;
            // Init offscreen canvas untuk kirim bersih ke server
            offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width  = canvas.width;
            offscreenCanvas.height = canvas.height;

            cameraActive = true; paused = false;
            cameraStatus.innerText = "🟢 Live";
            btnPause.innerText = "⏸ Pause";
            placeholder.style.display = "none";
            video.style.display = "none";   // disembunyikan, canvas yang tampil sebagai mirror
            canvas.style.display = "block";
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = handleWebSocketMessage;
            ws.onopen = () => {
                animationId = requestAnimationFrame(renderLoop); // render 60fps
                sendFrameToServer();                              // send throttled
            };
            ws.onclose = () => console.log("ws closed");
        } catch(e) { alert("Kamera tidak diizinkan atau error"); }
    }

    function pauseCamera() {
        if (!cameraActive) return;
        paused = !paused;
        if (paused) {
            // Render loop tetap jalan (video masih tampil + box tetap overlay)
            // Hanya pengiriman ke server yang dihentikan
            cameraStatus.innerText = "🟡 Paused";
            btnPause.innerText = "▶ Resume";
        } else {
            lastSendTime = 0; // kirim segera setelah resume
            cameraStatus.innerText = "🟢 Live";
            btnPause.innerText = "⏸ Pause";
        }
    }

    function stopCamera() {
        if(stream) { stream.getTracks().forEach(t => t.stop()); stream=null; }
        if(ws) ws.close();
        if(animationId) { cancelAnimationFrame(animationId); animationId=null; }
        cameraActive=false; paused=false; lastBox=null; targetBox=null; smoothBox=null;
        video.style.display="none"; canvas.style.display="none"; placeholder.style.display="flex";
        cameraStatus.innerText="🔴 Offline";
        detectInfo.innerText="Ready";
        btnPause.innerText="⏸ Pause";
    }

    async function uploadImage() {
        const input = document.createElement('input');
        input.type='file'; input.accept='image/*';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if(!file) return;
            const formData = new FormData();
            formData.append('file', file);
            const resp = await fetch('/upload', { method:'POST', body:formData });
            if(resp.ok) {
                const result = await resp.json();
                updateUI(result);
                const img = new Image();
                img.onload = () => {
                    // Sesuaikan ukuran canvas dengan gambar asli
                    canvas.width  = img.naturalWidth;
                    canvas.height = img.naturalHeight;
                    ctx.drawImage(img, 0, 0);
                    if(result.box) {
                        // Koordinat box sudah dalam ruang gambar asli — scaling 1:1
                        lastBox = {
                            box: result.box,
                            color: result.box_color,
                            label: `${result.skin_type} (${result.confidence}%)`,
                            srcW: img.naturalWidth,
                            srcH: img.naturalHeight
                        };
                        redrawBoxOnly();
                    }
                    video.style.display="none"; canvas.style.display="block"; placeholder.style.display="none";
                };
                img.src = URL.createObjectURL(file);
            } else alert("Tidak ada wajah terdeteksi");
        };
        input.click();
    }

    function clearData() {
        confidenceSpan.innerText="0%";
        skinConditionSpan.innerText="—";
        dateSpan.innerText="-"; timeSpan.innerText="-";
        [moistureBar,oilBar,acneBar,blackspotBar,wrinkleBar].forEach(b=>b.value=0);
        [moistureVal,oilVal,acneVal,blackspotVal,wrinkleVal].forEach(v=>v.innerText="0%");
        recommendationDiv.innerText="Belum ada rekomendasi";
        routineDiv.innerText="—";
        historyList=[];
        lastBox=null; targetBox=null; smoothBox=null; lastAICallTime=0; lastAISkinType=null;
        if(!cameraActive) { canvas.style.display="none"; placeholder.style.display="flex"; ctx.clearRect(0,0,canvas.width,canvas.height); }
        alert("Data cleared");
    }

    function showHistory() {
        if(historyList.length===0) return alert("Belum ada history");
        const modal=document.getElementById('historyModal');
        document.getElementById('historyList').innerText=historyList.join('\\n');
        modal.style.display="flex";
        document.querySelector('#historyModal .close').onclick=()=>modal.style.display="none";
    }

    let chart=null;
    function showResultAnalysis() {
        if(!currentResult) return alert("Belum ada hasil deteksi");
        const modal=document.getElementById('chartModal');
        modal.style.display="flex";
        const canvasChart=document.getElementById('resultChart');
        if(chart) chart.destroy();
        chart = new Chart(canvasChart, { type:'bar', data:{ labels:['Moisture','Sebum','Acne','Blackspot','Wrinkle'], datasets:[{ label:'Skor (%)', data:[currentResult.moisture,currentResult.oil,currentResult.acne,currentResult.blackspot,currentResult.wrinkle], backgroundColor:'#F45D9C', borderRadius:8 }] }, options:{ scales:{ y:{ beginAtZero:true, max:100 } }, responsive:true, maintainAspectRatio:true } });
        document.querySelector('#chartModal .close-chart').onclick=()=>modal.style.display="none";
    }

    async function downloadPDF() {
        if(!currentResult) return alert("Tidak ada data untuk diunduh");
        const name=nameInput.value||"-", age=ageInput.value||"-";
        const rec=recommendationDiv.innerText, routine=routineDiv.innerText;
        const url=`/pdf?name=${encodeURIComponent(name)}&age=${encodeURIComponent(age)}&skin_type=${currentResult.skin_type}&confidence=${currentResult.confidence}&moisture=${currentResult.moisture}&oil=${currentResult.oil}&acne=${currentResult.acne}&blackspot=${currentResult.blackspot}&wrinkle=${currentResult.wrinkle}&date=${currentResult.date}&time=${currentResult.time}&recommendation=${encodeURIComponent(rec)}&routine=${encodeURIComponent(routine)}`;
        detectInfo.innerText = "📄 Membuat PDF (termasuk foto wajah)...";
        window.open(url,'_blank');
        setTimeout(()=>{ detectInfo.innerText = "✅ PDF berhasil dibuat!"; }, 1200);
    }

    function toggleDarkMode() {
        document.body.classList.toggle('dark');
        btnDark.innerText = document.body.classList.contains('dark') ? "☀ Light Mode" : "🌙 Dark Mode";
    }

    function setActive(btn) { document.querySelectorAll('.sidebar-btn').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); }
    btnOpen.onclick = () => { setActive(btnOpen); startCamera(); };
    btnPause.onclick = () => { setActive(btnPause); pauseCamera(); };
    btnClose.onclick = () => { setActive(btnClose); stopCamera(); };
    btnUpload.onclick = () => { setActive(btnUpload); uploadImage(); };
    btnClear.onclick = () => { setActive(btnClear); clearData(); };
    btnHistory.onclick = () => { setActive(btnHistory); showHistory(); };
    btnResult.onclick = () => { setActive(btnResult); showResultAnalysis(); };
    btnPdf.onclick = () => { setActive(btnPdf); downloadPDF(); };
    btnDark.onclick = () => { setActive(btnDark); toggleDarkMode(); };
</script>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_bytes = await websocket.receive_bytes()
            result = await asyncio.to_thread(process_frame, frame_bytes)
            if result and result[0]:
                detection_data, _ = result
                await websocket.send_json(detection_data)
            else:
                await websocket.send_json({"error": "No face detected"})
    except WebSocketDisconnect:
        pass

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    result = process_frame(contents)
    if result is None or result[0] is None:
        return JSONResponse({"error": "No face detected or invalid image"}, status_code=400)
    detection_data, _ = result
    # Simpan juga image bytes asli untuk PDF (override latest_frame_store dari process_frame)
    latest_frame_store["image_bytes"] = contents
    return JSONResponse(detection_data)

@app.post("/recommendations")
async def get_ai_recommendations(request: Request):
    data = await request.json()
    skin_type = data.get("skin_type")
    confidence = data.get("confidence")
    moisture = data.get("moisture")
    oil = data.get("oil")
    acne = data.get("acne")
    blackspot = data.get("blackspot")
    wrinkle = data.get("wrinkle")
    name = data.get("name", "User")
    age = data.get("age", "tidak diketahui")

    prompt = f"""Kamu adalah dermatologis AI. Berikan rekomendasi perawatan kulit berdasarkan data analisis berikut:

Nama: {name}, Umur: {age} tahun
Tipe Kulit: {skin_type} (confidence: {confidence}%)
Moisture: {moisture}%, Sebum/Oil: {oil}%, Acne: {acne}%, Blackspot: {blackspot}%, Wrinkle: {wrinkle}%

Balas HANYA dalam format JSON berikut (tanpa markdown, tanpa penjelasan lain):
{{
  "rec": "• produk/bahan aktif 1\\n• produk/bahan aktif 2\\n• produk/bahan aktif 3\\n• produk/bahan aktif 4\\n• produk/bahan aktif 5",
  "routine": "☀️ Pagi: langkah1 → langkah2 → langkah3 → SPF\\n🌙 Malam: langkah1 → langkah2 → langkah3"
}}"""

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                GEMINI_API_URL,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 8000  # ditingkatkan agar respons tidak terpotong
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            text = result["candidates"][0]["content"]["parts"][0]["text"]

            # Bersihkan markdown
            text = text.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            text = text.strip()

            # Coba parse langsung
            try:
                ai_data = json.loads(text)
                logging.info("Gemini AI berhasil memberikan rekomendasi")
                return JSONResponse(ai_data)
            except json.JSONDecodeError:
                # Gunakan json_repair untuk memperbaiki JSON yang mungkin terpotong
                try:
                    repaired = repair_json(text)
                    ai_data = json.loads(repaired)
                    logging.info("Gemini AI berhasil memberikan rekomendasi (diperbaiki dengan json_repair)")
                    return JSONResponse(ai_data)
                except Exception as repair_e:
                    logging.error(f"Gagal memperbaiki JSON dengan json_repair: {repair_e}")
                    # Coba ekstrak dengan regex sebagai fallback terakhir
                    match = re.search(r'\{.*\}', text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                        try:
                            ai_data = json.loads(json_str)
                            logging.info("Gemini AI berhasil memberikan rekomendasi (diekstrak dengan regex)")
                            return JSONResponse(ai_data)
                        except json.JSONDecodeError as e:
                            logging.error(f"Gagal parse JSON hasil ekstrak: {e}")
                    else:
                        logging.error(f"Tidak ditemukan JSON dalam respons: {text[:500]}")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logging.error("QUOTA LIMIT atau terlalu banyak request ke Gemini API")
            else:
                logging.error(f"HTTP error {e.response.status_code} saat memanggil Gemini: {e.response.text}")
        except httpx.TimeoutException:
            logging.error("Timeout saat memanggil Gemini API")
        except Exception as e:
            logging.error(f"Error tak terduga saat memanggil Gemini: {str(e)}")
    
    fallback = get_static_recommendation(skin_type)
    logging.warning(f"Menggunakan rekomendasi statis untuk skin_type={skin_type}")
    return JSONResponse(fallback)

@app.post("/clear")
async def clear_data():
    return {"status": "cleared"}

@app.get("/history")
async def get_history():
    return {"history": []}

@app.get("/pdf")
async def generate_pdf(
    name: str = "",
    age: str = "",
    skin_type: str = "",
    confidence: int = 0,
    moisture: int = 0,
    oil: int = 0,
    acne: int = 0,
    blackspot: int = 0,
    wrinkle: int = 0,
    date: str = "",
    time: str = "",
    recommendation: str = "",
    routine: str = ""
):
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.lib.colors import HexColor, white, black
    import io

    # ── Warna palette skincare premium ──
    PINK       = HexColor("#F45D9C")
    PINK_LIGHT = HexColor("#FFD6EA")
    PINK_DARK  = HexColor("#C93E7D")
    ORANGE     = HexColor("#FF9F3D")
    BG_CREAM   = HexColor("#FFF5F9")
    GRAY_TEXT  = HexColor("#6B6B80")
    DARK_TEXT  = HexColor("#2D2D3A")
    CARD_BG    = HexColor("#FFFFFF")
    PROGRESS_BG = HexColor("#F0EEF5")

    pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
    W, H = A4  # 595 x 842

    c = rl_canvas.Canvas(pdf_path, pagesize=A4)

    # ─────────────────────────────────────────
    # Helper: rounded rectangle
    # ─────────────────────────────────────────
    def rounded_rect(x, y, w, h, r, fill_color=None, stroke_color=None, stroke_width=1):
        if fill_color:
            c.setFillColor(fill_color)
        if stroke_color:
            c.setStrokeColor(stroke_color)
            c.setLineWidth(stroke_width)
        else:
            c.setStrokeColor(HexColor("#00000000"))
        p = c.beginPath()
        p.moveTo(x + r, y)
        p.lineTo(x + w - r, y)
        p.arcTo(x + w - 2*r, y, x + w, y + 2*r, startAng=270, extent=90)
        p.lineTo(x + w, y + h - r)
        p.arcTo(x + w - 2*r, y + h - 2*r, x + w, y + h, startAng=0, extent=90)
        p.lineTo(x + r, y + h)
        p.arcTo(x, y + h - 2*r, x + 2*r, y + h, startAng=90, extent=90)
        p.lineTo(x, y + r)
        p.arcTo(x, y, x + 2*r, y + 2*r, startAng=180, extent=90)
        p.close()
        c.drawPath(p, fill=1 if fill_color else 0, stroke=1 if stroke_color else 0)

    # ─────────────────────────────────────────
    # PAGE 1 ── Header & Profile
    # ─────────────────────────────────────────

    # Background cream
    c.setFillColor(BG_CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Header gradient bar (simulate with two rects)
    c.setFillColor(PINK_DARK)
    c.rect(0, H - 110, W, 110, fill=1, stroke=0)
    c.setFillColor(PINK)
    c.rect(0, H - 110, W * 0.65, 110, fill=1, stroke=0)

    # Decorative circles in header
    c.setFillColor(HexColor("#FFFFFF20"))
    c.circle(W - 60, H - 30, 70, fill=1, stroke=0)
    c.setFillColor(HexColor("#FFFFFF15"))
    c.circle(W - 20, H - 80, 50, fill=1, stroke=0)

    # Logo / Brand name
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(36, H - 52, "Facial Skin Detection")
    c.setFont("Helvetica", 11)
    c.setFillColor(PINK_LIGHT)
    c.drawString(36, H - 70, "Smart Skincare Analysis Report")

    # Date/time badge on header right
    c.setFillColor(HexColor("#FFFFFF25"))
    rounded_rect(W - 200, H - 90, 165, 36, 8, fill_color=HexColor("#FFFFFF25"))
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(W - 192, H - 66, f"  {date}   {time}")
    c.setFont("Helvetica", 8)
    c.drawString(W - 192, H - 79, "  Analysis Date & Time")

    # ── Patient Info Card ──
    card_y = H - 210
    rounded_rect(30, card_y, W - 60, 88, 12, fill_color=CARD_BG, stroke_color=PINK_LIGHT, stroke_width=1.5)

    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, card_y + 62, "Patient Profile")

    # Name & Age columns
    c.setFillColor(GRAY_TEXT)
    c.setFont("Helvetica", 9)
    c.drawString(50, card_y + 44, "FULL NAME")
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(DARK_TEXT)
    c.drawString(50, card_y + 28, name or "—")

    c.setFillColor(GRAY_TEXT)
    c.setFont("Helvetica", 9)
    c.drawString(250, card_y + 44, "AGE")
    c.setFillColor(DARK_TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(250, card_y + 28, f"{age} years old" if age and age != "-" else "—")

    c.setFillColor(GRAY_TEXT)
    c.setFont("Helvetica", 9)
    c.drawString(400, card_y + 44, "SKIN TYPE")
    c.setFillColor(PINK_DARK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(400, card_y + 28, skin_type or "—")

    # ── Confidence Score Hero ──
    hero_y = H - 340
    # Left: big score circle
    cx_circle = 100
    cy_circle = hero_y + 55
    c.setFillColor(PINK_LIGHT)
    c.circle(cx_circle, cy_circle, 52, fill=1, stroke=0)
    c.setFillColor(PINK)
    c.circle(cx_circle, cy_circle, 44, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    score_txt = f"{confidence}%"
    c.drawCentredString(cx_circle, cy_circle + 2, score_txt)
    c.setFont("Helvetica", 8)
    c.drawCentredString(cx_circle, cy_circle - 14, "Confidence")

    # Right: skin type label + description
    c.setFillColor(DARK_TEXT)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(175, hero_y + 75, skin_type or "—")
    c.setFillColor(GRAY_TEXT)
    c.setFont("Helvetica", 10)
    descriptions = {
        "Acne": "Your skin shows signs of acne-prone condition.",
        "Dry": "Your skin tends to be dry and needs extra hydration.",
        "Oily": "Your skin produces excess sebum/oil.",
        "Normal": "Your skin is in a well-balanced condition."
    }
    c.drawString(175, hero_y + 58, descriptions.get(skin_type, "Analysis complete."))

    # ── Skin Parameter Progress Bars ──
    section_y = H - 480
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(36, section_y + 8, "Skin Parameter Analysis")

    params = [
        ("Moisture", moisture, "#36B5FF"),
        ("Sebum / Oil", oil, "#FF9F3D"),
        ("Acne Level", acne, "#F45D9C"),
        ("Blackspot", blackspot, "#9C6FDB"),
        ("Wrinkle", wrinkle, "#FF6B6B"),
    ]

    bar_start_y = section_y - 18
    bar_w_full  = (W - 72)
    col_w       = bar_w_full / 3  # 3 columns
    for i, (label, val, hex_color) in enumerate(params):
        col  = i % 3
        row  = i // 3
        bx   = 36 + col * col_w
        by   = bar_start_y - row * 62

        # Card background
        rounded_rect(bx + 2, by - 36, col_w - 8, 56, 8, fill_color=CARD_BG)

        # Label
        c.setFillColor(GRAY_TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(bx + 12, by + 14, label.upper())

        # Value
        c.setFillColor(HexColor(hex_color))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(bx + 12, by - 4, f"{val}%")

        # Progress bar track
        bar_track_w = col_w - 28
        c.setFillColor(PROGRESS_BG)
        rounded_rect(bx + 12, by - 24, bar_track_w, 8, 4, fill_color=PROGRESS_BG)

        # Progress bar fill
        fill_w = max(6, int(bar_track_w * val / 100))
        c.setFillColor(HexColor(hex_color))
        rounded_rect(bx + 12, by - 24, fill_w, 8, 4, fill_color=HexColor(hex_color))

    # ── Divider ──
    divider_y = section_y - 150
    c.setStrokeColor(PINK_LIGHT)
    c.setLineWidth(1)
    c.line(36, divider_y, W - 36, divider_y)

    # ── Word-wrap helper ──
    def wrap_text(text, font_name, font_size, max_width):
        """Split text into lines that fit within max_width using pdfgen string width."""
        from reportlab.pdfbase.pdfmetrics import stringWidth
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if stringWidth(test, font_name, font_size) <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    FONT_REC  = "Helvetica"
    FONT_SIZE = 9
    BOX_PAD_L = 16   # left padding inside box
    BOX_PAD_R = 12   # right padding inside box
    LINE_H    = 14   # vertical spacing per rendered line

    half_w = (W - 72) / 2 - 6          # width of each half-box
    inner_w = half_w - BOX_PAD_L - BOX_PAD_R  # usable text width inside box

    # ── Pre-render rec lines with wrapping ──
    rec_lines_raw = [l.strip() for l in recommendation.replace("\\n", "\n").split("\n") if l.strip()]
    rec_rendered = []
    for line in rec_lines_raw[:8]:
        clean = line.lstrip("•-– ").strip()
        prefix = "✦  "
        wrapped = wrap_text(prefix + clean, FONT_REC, FONT_SIZE, inner_w)
        rec_rendered.extend(wrapped)

    # ── Pre-render routine lines with wrapping ──
    routine_lines_raw = [l.strip() for l in routine.replace("\\n", "\n").split("\n") if l.strip()]
    routine_rendered = []
    for line in routine_lines_raw[:8]:
        wrapped = wrap_text(line, FONT_REC, FONT_SIZE, inner_w)
        routine_rendered.extend(wrapped)

    # Box height driven by whichever side has more lines
    max_lines   = max(len(rec_rendered), len(routine_rendered), 5)
    rec_box_h   = max_lines * LINE_H + 28

    # ── Recommendation Section ──
    rec_y = divider_y - 22
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(36, rec_y, "Smart Skincare Recommendation")

    rounded_rect(36, rec_y - rec_box_h - 10, half_w, rec_box_h, 10,
                 fill_color=HexColor("#FFF0F7"), stroke_color=PINK_LIGHT, stroke_width=1)

    c.setFillColor(DARK_TEXT)
    c.setFont(FONT_REC, FONT_SIZE)
    for idx, line in enumerate(rec_rendered):
        c.drawString(36 + BOX_PAD_L, rec_y - 28 - idx * LINE_H, line)

    # ── Routine Section ──
    rx = 36 + half_w + 6
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(rx, rec_y, "Daily Care Routine")

    rounded_rect(rx, rec_y - rec_box_h - 10, half_w, rec_box_h, 10,
                 fill_color=HexColor("#FFF8F0"), stroke_color=HexColor("#FFD6A0"), stroke_width=1)

    c.setFillColor(DARK_TEXT)
    c.setFont(FONT_REC, FONT_SIZE)
    for idx, line in enumerate(routine_rendered):
        c.drawString(rx + BOX_PAD_L, rec_y - 28 - idx * LINE_H, line)

    # ── Footer ──
    footer_y = 28
    c.setFillColor(PINK)
    c.rect(0, 0, W, footer_y + 10, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, footer_y - 4, "Generated by Facial Skin Detection  •  For informational purposes only  •  Consult a dermatologist for medical advice")

    # ─────────────────────────────────────────
    # PAGE 2 ── Bar Chart
    # ─────────────────────────────────────────
    c.showPage()

    # Background
    c.setFillColor(BG_CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Header strip
    c.setFillColor(PINK)
    c.rect(0, H - 70, W, 70, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, H - 42, "Skin Analysis Chart")
    c.setFont("Helvetica", 10)
    c.setFillColor(PINK_LIGHT)
    c.drawString(36, H - 58, f"{name}  |  {date}  {time}")

    # Generate chart with matplotlib
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor('#FFF5F9')
    ax.set_facecolor('#FFF5F9')

    param_labels = ["Moisture", "Sebum", "Acne", "Blackspot", "Wrinkle"]
    param_values = [moisture, oil, acne, blackspot, wrinkle]
    bar_colors   = ["#36B5FF", "#FF9F3D", "#F45D9C", "#9C6FDB", "#FF6B6B"]

    bars = ax.bar(param_labels, param_values, color=bar_colors, width=0.55, zorder=3)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_tick_params(labelsize=9, colors='#6B6B80')
    ax.xaxis.set_tick_params(labelsize=10, colors='#2D2D3A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#F0EEF5')
    ax.spines['bottom'].set_color('#F0EEF5')
    ax.yaxis.grid(True, color='#F0EEF5', linewidth=1.2, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Score (%)", fontsize=9, color='#6B6B80')
    ax.set_title(f"Skin Parameters — {skin_type} ({confidence}% confidence)",
                 fontsize=12, color='#F45D9C', fontweight='bold', pad=12)

    for bar, val in zip(bars, param_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val}%", ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#2D2D3A')

    plt.tight_layout(pad=1.5)

    chart_buf = io.BytesIO()
    plt.savefig(chart_buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#FFF5F9', edgecolor='none')
    plt.close()
    chart_buf.seek(0)

    chart_img = ImageReader(chart_buf)
    chart_w   = W - 72
    chart_h   = chart_w * 4.2 / 7.5
    c.drawImage(chart_img, 36, H - 110 - chart_h, width=chart_w, height=chart_h,
                preserveAspectRatio=True, mask='auto')

    # ── Metric summary cards below chart ──
    summary_y = H - 110 - chart_h - 50
    c.setFillColor(DARK_TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, summary_y + 12, "Summary")

    card_data = [
        ("Moisture", moisture, "#36B5FF",
         "Good" if moisture >= 70 else "Low"),
        ("Sebum",    oil,      "#FF9F3D",
         "High" if oil >= 70 else "Normal"),
        ("Acne",     acne,     "#F45D9C",
         "Severe" if acne >= 60 else ("Moderate" if acne >= 30 else "Mild")),
        ("Blackspot",blackspot,"#9C6FDB",
         "Visible" if blackspot >= 40 else "Minimal"),
        ("Wrinkle",  wrinkle,  "#FF6B6B",
         "Prominent" if wrinkle >= 50 else "Subtle"),
    ]

    card_total_w = W - 72
    cw = card_total_w / 5
    for i, (lbl, val, hex_c, status) in enumerate(card_data):
        cx_ = 36 + i * cw
        cy_ = summary_y - 56
        rounded_rect(cx_ + 3, cy_, cw - 6, 52, 8,
                     fill_color=CARD_BG, stroke_color=HexColor(hex_c + "55"), stroke_width=1)
        c.setFillColor(HexColor(hex_c))
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(cx_ + cw / 2, cy_ + 28, f"{val}%")
        c.setFillColor(GRAY_TEXT)
        c.setFont("Helvetica", 8)
        c.drawCentredString(cx_ + cw / 2, cy_ + 14, lbl)
        c.setFillColor(DARK_TEXT)
        c.setFont("Helvetica-Bold", 8)
        c.drawCentredString(cx_ + cw / 2, cy_ + 4, status)

    # ── Footer page 2 ──
    c.setFillColor(PINK)
    c.rect(0, 0, W, 38, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, 14, "Generated by Facial Skin Detection  •  For informational purposes only  •  Consult a dermatologist for medical advice")

    # ─────────────────────────────────────────
    # PAGE 3 ── Line Graph & Pie Chart
    # ─────────────────────────────────────────
    c.showPage()

    # Background
    c.setFillColor(BG_CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # Header strip
    c.setFillColor(PINK)
    c.rect(0, H - 70, W, 70, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(36, H - 42, "Skin Health Overview")
    c.setFont("Helvetica", 10)
    c.setFillColor(PINK_LIGHT)
    c.drawString(36, H - 58, f"{name}  |  {date}  {time}")

    param_labels = ["Moisture", "Sebum", "Acne", "Blackspot", "Wrinkle"]
    param_values = [moisture, oil, acne, blackspot, wrinkle]
    bar_colors   = ["#36B5FF", "#FF9F3D", "#F45D9C", "#9C6FDB", "#FF6B6B"]

    # ── Line Graph (top half) ──
    fig_line, ax_line = plt.subplots(figsize=(7.5, 3.5))
    fig_line.patch.set_facecolor('#FFF5F9')
    ax_line.set_facecolor('#FFF5F9')

    x_pos = list(range(len(param_labels)))
    ax_line.plot(x_pos, param_values, color='#F45D9C', linewidth=2.5,
                 marker='o', markersize=9, markerfacecolor='white',
                 markeredgecolor='#F45D9C', markeredgewidth=2.5, zorder=3)

    # Fill area under line
    ax_line.fill_between(x_pos, param_values, alpha=0.12, color='#F45D9C')

    # Annotate each point
    for xi, yi in zip(x_pos, param_values):
        ax_line.annotate(f"{yi}%", (xi, yi),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=10, fontweight='bold', color='#2D2D3A')

    ax_line.set_xticks(x_pos)
    ax_line.set_xticklabels(param_labels, fontsize=10, color='#2D2D3A')
    ax_line.set_ylim(0, 115)
    ax_line.set_yticks(range(0, 101, 20))
    ax_line.yaxis.set_tick_params(labelsize=9, colors='#6B6B80')
    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['left'].set_color('#F0EEF5')
    ax_line.spines['bottom'].set_color('#F0EEF5')
    ax_line.yaxis.grid(True, color='#F0EEF5', linewidth=1.2, zorder=0)
    ax_line.set_axisbelow(True)
    ax_line.set_ylabel("Score (%)", fontsize=9, color='#6B6B80')
    ax_line.set_title(f"Skin Parameter Trend — {skin_type} ({confidence}% confidence)",
                      fontsize=12, color='#F45D9C', fontweight='bold', pad=12)
    plt.tight_layout(pad=1.5)

    line_buf = io.BytesIO()
    plt.savefig(line_buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#FFF5F9', edgecolor='none')
    plt.close()
    line_buf.seek(0)

    line_img = ImageReader(line_buf)
    line_w = W - 72
    line_h = line_w * 3.5 / 7.5
    line_y = H - 90 - line_h
    c.drawImage(line_img, 36, line_y, width=line_w, height=line_h,
                preserveAspectRatio=True, mask='auto')

    # Section label
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, line_y - 22, "Line Graph — Skin Parameter Trend")

    # ── Pie Chart (bottom half) ──
    fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
    fig_pie.patch.set_facecolor('#FFF5F9')
    ax_pie.set_facecolor('#FFF5F9')

    # Explode the largest slice slightly
    max_idx = param_values.index(max(param_values))
    explode = [0.05 if i == max_idx else 0 for i in range(len(param_values))]

    wedges, texts, autotexts = ax_pie.pie(
        param_values,
        labels=param_labels,
        colors=bar_colors,
        autopct='%1.1f%%',
        startangle=140,
        explode=explode,
        pctdistance=0.78,
        wedgeprops=dict(linewidth=1.5, edgecolor='white')
    )
    for text in texts:
        text.set_fontsize(10)
        text.set_color('#2D2D3A')
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax_pie.set_title(f"Skin Parameter Distribution — {skin_type}",
                     fontsize=12, color='#F45D9C', fontweight='bold', pad=14)
    plt.tight_layout(pad=1.5)

    pie_buf = io.BytesIO()
    plt.savefig(pie_buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#FFF5F9', edgecolor='none')
    plt.close()
    pie_buf.seek(0)

    pie_img  = ImageReader(pie_buf)
    pie_w    = W - 72
    pie_h    = pie_w * 4.0 / 6.0
    pie_y    = line_y - 40 - pie_h
    c.drawImage(pie_img, 36, pie_y, width=pie_w, height=pie_h,
                preserveAspectRatio=True, mask='auto')

    # Section label
    c.setFillColor(PINK)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(36, pie_y - 20, "Pie Chart — Parameter Composition")

    # ── Footer page 3 ──
    c.setFillColor(PINK)
    c.rect(0, 0, W, 38, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica", 8)
    c.drawCentredString(W / 2, 14, "Generated by Facial Skin Detection  •  For informational purposes only  •  Consult a dermatologist for medical advice")

    # ─────────────────────────────────────────
    # PAGE 4 ── User Photo with Detection Box
    # ─────────────────────────────────────────
    frame_data = latest_frame_store.get("image_bytes")
    if frame_data:
        c.showPage()

        # Background
        c.setFillColor(BG_CREAM)
        c.rect(0, 0, W, H, fill=1, stroke=0)

        # Header strip
        c.setFillColor(PINK)
        c.rect(0, H - 70, W, 70, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(36, H - 42, "Foto Analisis Wajah")
        c.setFont("Helvetica", 10)
        c.setFillColor(PINK_LIGHT)
        c.drawString(36, H - 58, f"{name}  |  {date}  {time}")

        try:
            # Decode gambar dari bytes
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_img is not None:
                # Gambar bounding box ke frame
                stored_box   = latest_frame_store.get("box")
                stored_color = latest_frame_store.get("box_color", "#F45D9C")
                stored_label = f"{latest_frame_store.get('skin_type', skin_type)} ({latest_frame_store.get('confidence', confidence)}%)"

                if stored_box:
                    x1b, y1b, x2b, y2b = stored_box

                    # Konversi hex color ke BGR untuk cv2
                    hex_c = stored_color.lstrip("#")
                    r_c, g_c, b_c = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
                    bgr_color = (b_c, g_c, r_c)

                    # Gambar rectangle
                    cv2.rectangle(frame_img, (x1b, y1b), (x2b, y2b), bgr_color, 3)

                    # Label background
                    label_font  = cv2.FONT_HERSHEY_SIMPLEX
                    label_scale = 0.7
                    label_thick = 2
                    (tw, th), _ = cv2.getTextSize(stored_label, label_font, label_scale, label_thick)
                    cv2.rectangle(frame_img, (x1b, y1b - th - 14), (x1b + tw + 10, y1b), bgr_color, -1)
                    cv2.putText(frame_img, stored_label,
                                (x1b + 5, y1b - 7),
                                label_font, label_scale, (255,255,255), label_thick, cv2.LINE_AA)

                # Encode ke PNG untuk ReportLab
                _, png_buf = cv2.imencode(".png", frame_img)
                img_bytes_io = io.BytesIO(png_buf.tobytes())
                user_img_reader = ImageReader(img_bytes_io)

                # Hitung ukuran gambar agar muat di halaman dengan margin
                img_h_orig, img_w_orig = frame_img.shape[:2]
                max_img_w = W - 72
                max_img_h = H - 130  # sisakan ruang header + keterangan + footer
                scale = min(max_img_w / img_w_orig, max_img_h / img_h_orig)
                draw_w = img_w_orig * scale
                draw_h = img_h_orig * scale
                draw_x = (W - draw_w) / 2
                draw_y = H - 90 - draw_h

                # Shadow / border card
                rounded_rect(draw_x - 6, draw_y - 6, draw_w + 12, draw_h + 12, 10,
                             fill_color=PINK_LIGHT, stroke_color=None)

                # Gambar foto
                c.drawImage(user_img_reader, draw_x, draw_y, width=draw_w, height=draw_h,
                            preserveAspectRatio=True, mask='auto')

                # Keterangan di bawah foto
                caption_y = draw_y - 30
                c.setFillColor(PINK)
                c.setFont("Helvetica-Bold", 11)
                c.drawCentredString(W / 2, caption_y,
                    f"Deteksi: {latest_frame_store.get('skin_type', skin_type)}  "
                    f"({latest_frame_store.get('confidence', confidence)}% confidence)")
                c.setFillColor(GRAY_TEXT)
                c.setFont("Helvetica", 9)
                if stored_box:
                    c.drawCentredString(W / 2, caption_y - 16,
                        f"Bounding box: [{stored_box[0]}, {stored_box[1]}, {stored_box[2]}, {stored_box[3]}]  —  "
                        f"Warna: {stored_color}")

        except Exception as e:
            logging.error(f"Gagal menyisipkan foto ke PDF: {e}")
            c.setFillColor(GRAY_TEXT)
            c.setFont("Helvetica", 11)
            c.drawCentredString(W / 2, H / 2, "Foto tidak tersedia")

        # ── Footer page 4 ──
        c.setFillColor(PINK)
        c.rect(0, 0, W, 38, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica", 8)
        c.drawCentredString(W / 2, 14, "Generated by Facial Skin Detection  •  For informational purposes only  •  Consult a dermatologist for medical advice")

    c.save()

    return FileResponse(pdf_path, media_type="application/pdf", filename=f"skin_report_{name or 'user'}.pdf")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
