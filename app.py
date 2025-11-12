# app.py
import os
import time
import uuid
import json
import threading
import traceback
import gc
from pathlib import Path
from typing import Dict, Any

from flask import Flask, request, jsonify, send_file, abort
from flask_sock import Sock
import torch
import numpy as np
from PIL import Image
import imageio

# -------------------------
# IMPORT YOUR COMFYUI NODES
# -------------------------
# Ensure your comfy repo is on PYTHONPATH or same dir as this app.
# This mirrors the imports in your Colab script.
import sys
# If your ComfyUI is placed at /app/ComfyUI or ./ComfyUI, adjust as needed:
COMFY_ROOT = os.environ.get("COMFY_ROOT", "/content/ComfyUI")
if COMFY_ROOT not in sys.path:
    sys.path.insert(0, COMFY_ROOT)

from comfy import model_management

from nodes import (
    CheckpointLoaderSimple,
    CLIPLoader,
    CLIPTextEncode,
    VAEDecode,
    VAELoader,
    KSampler,
    UNETLoader
)

from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_model_advanced import ModelSamplingSD3
from comfy_extras.nodes_hunyuan import EmptyHunyuanLatentVideo
from comfy_extras.nodes_images import SaveAnimatedWEBP
from comfy_extras.nodes_video import SaveWEBM

# -------------------------
# SERVER & JOB MANAGEMENT
# -------------------------
app = Flask(__name__, static_folder=None)
sock = Sock(app)

JOBS: Dict[str, Dict[str, Any]] = {}  # job_id -> {status, progress, out, logs, error}
WS_CLIENTS = {}  # job_id -> list of websocket connections

# Where to save outputs
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/comfy_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# MODEL/COMPONENT PLACEHOLDERS
# -------------------------
# We will load the same objects you used in Colab; keep names familiar.
useQ6 = False  # change via env var if desired

# Node instances (similar to your notebook)
unet_loader = None
clip_loader = None
clip_encode_positive = None
clip_encode_negative = None
vae_loader = None
empty_latent_video = None
ksampler = None
vae_decode = None
save_webp = None
save_webm = None

# The actual loaded model objects (to be populated at startup)
LOADED = {
    "clip": None,
    "unet": None,
    "vae": None
}

# -------------------------
# UTILS (unchanged logic)
# -------------------------
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    # do not try to wipe globals here in server context

def save_as_mp4(images, filename_prefix, fps, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.mp4"

    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webp(images, filename_prefix, fps, quality=90, lossless=False, method=4, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webp"
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'lossless': bool(lossless),
        'method': int(method)
    }

    with imageio.get_writer(output_path, format='WEBP', mode='I', **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_webm(images, filename_prefix, fps, codec="vp9", quality=32, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.webm"
    frames = [(img.cpu().numpy() * 255).astype(np.uint8) for img in images]

    kwargs = {
        'fps': int(fps),
        'quality': int(quality),
        'codec': str(codec),
        'output_params': ['-crf', str(int(quality))]
    }

    with imageio.get_writer(output_path, format='FFMPEG', mode='I', **kwargs) as writer:
        for frame in frames:
            writer.append_data(frame)

    return output_path

def save_as_image(image, filename_prefix, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{filename_prefix}.png"
    frame = (image.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(frame).save(output_path)
    return output_path

# -------------------------
# CORE: generate_video (adapted to server)
# Keep the same internal steps; removed display() and made it return output_path.
# -------------------------
def generate_video_sync(
    positive_prompt: str = "a fox moving quickly in a beautiful winter scenery nature trees mountains daytime tracking camera",
    negative_prompt: str = "Bright tones, overexposure, static, blurry details, subtitles, artistic style, artwork, painting, still image, dull overall, worst quality, low quality, JPEG compression artifacts, ugly, deformed, extra fingers, poorly drawn hands, poorly drawn face, disfigured, malformed limbs, finger fusion, static frame, messy background, three legs, crowded background, walking backwards",
    width: int = 832,
    height: int = 480,
    seed: int = 82628696717253,
    steps: int = 30,
    cfg_scale: float = 1.0,
    sampler_name: str = "uni_pc",
    scheduler: str = "simple",
    frames: int = 33,
    fps: int = 16,
    output_format: str = "mp4",
    job_id: str = None,
    model_variant: str = None
):
    """
    This function preserves the same logical steps as your original Colab function:
    - load/use clip encoder
    - prepare prompts
    - create empty latents
    - load/unload unet (we'll use preloaded LOADED['unet'])
    - sample using ksampler
    - decode with VAE
    - save file to disk and return path
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    prefix = f"job_{job_id}"

    def _log(msg):
        # append to JOBS logs and broadcast over WS if present
        entry = f"[generate] {msg}"
        JOBS[job_id]["logs"].append(entry)
        JOBS[job_id]["last_log"] = entry
        broadcast_ws(job_id, {"type":"log", "text": entry})

    try:
        with torch.inference_mode():
            _log("Starting generation (inference mode).")
            _log("Encoding prompts with Text Encoder...")

            # Use the preloaded CLIP/text encoder if available, else attempt to load dynamically
            clip = LOADED.get("clip")
            if clip is None:
                _log("Clip encoder not loaded in memory; attempting to load (this may be slow).")
                clip = clip_loader.load_clip("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "wan", "default")[0]
            positive = clip_encode_positive.encode(clip, positive_prompt)[0]
            negative = clip_encode_negative.encode(clip, negative_prompt)[0]

            # free clip reference if not needed (but keep LOADED['clip'] if preloaded)
            if LOADED.get("clip") is None:
                del clip
                torch.cuda.empty_cache()
                gc.collect()

            _log("Generating empty latents...")
            empty_latent = empty_latent_video.generate(width, height, frames, 1)[0]

            _log("Using UNET model for sampling...")
            # use preloaded unet model (LOADED['unet']) - this was loaded at startup
            model = LOADED.get("unet")
            if model is None:
                _log("UNET not present in memory; loading ad-hoc (slower).")
                if useQ6:
                    model = unet_loader.load_unet("wan2.1-t2v-14b-Q6_K.gguf")[0]
                else:
                    model = unet_loader.load_unet("wan2.1-t2v-14b-Q5_0.gguf")[0]

            _log("Running sampler...")
            sampled = ksampler.sample(
                model=model,
                seed=seed,
                steps=steps,
                cfg=cfg_scale,
                sampler_name=sampler_name,
                scheduler=scheduler,
                positive=positive,
                negative=negative,
                latent_image=empty_latent
            )[0]

            # if we loaded a local 'model', we won't delete LOADED['unet'] if it exists,
            # but we do clear local only references if created ad-hoc
            if LOADED.get("unet") is None and model is not None:
                try:
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception:
                    pass

            _log("Loading VAE...")
            vae = LOADED.get("vae")
            if vae is None:
                vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]

            _log("Decoding latents to images...")
            decoded = vae_decode.decode(vae, sampled)[0]

            # save outputs
            output_path = ""
            if frames == 1:
                _log("Single frame -> saving PNG.")
                output_path = save_as_image(decoded[0], prefix)
            else:
                if output_format.lower() == "webm":
                    _log("Saving WEBM")
                    output_path = save_as_webm(decoded, prefix, fps=fps, codec="vp9", quality=10)
                elif output_format.lower() == "mp4":
                    _log("Saving MP4")
                    output_path = save_as_mp4(decoded, prefix, fps)
                else:
                    raise ValueError(f"Unsupported output format: {output_format}")

            _log(f"Saved output to {output_path}")
            # final cleanup
            clear_memory()
            return output_path

    except Exception as e:
        # bubble up error
        _log("Error in generate_video_sync: " + str(e))
        _log(traceback.format_exc())
        clear_memory()
        raise

# -------------------------
# MODEL LOADING (once at startup)
# -------------------------
def load_models_on_startup():
    global unet_loader, clip_loader, clip_encode_positive, clip_encode_negative
    global vae_loader, empty_latent_video, ksampler, vae_decode, save_webp, save_webm
    global LOADED

    # Instantiate node objects once (same as your notebook)
    unet_loader = UnetLoaderGGUF()
    clip_loader = CLIPLoader()
    clip_encode_positive = CLIPTextEncode()
    clip_encode_negative = CLIPTextEncode()
    vae_loader = VAELoader()
    empty_latent_video = EmptyHunyuanLatentVideo()
    ksampler = KSampler()
    vae_decode = VAEDecode()
    save_webp = SaveAnimatedWEBP()
    save_webm = SaveWEBM()

    # Load heavy models into LOADED dict (if files are present)
    try:
        # Attempt to load text encoder (clip)
        clip_path = os.environ.get("CLIP_PATH", "umt5_xxl_fp8_e4m3fn_scaled.safetensors")
        if Path(clip_path).exists():
            app.logger.info("Loading CLIP/text encoder into memory...")
            LOADED["clip"] = clip_loader.load_clip(clip_path, "wan", "default")[0]
        else:
            app.logger.warning(f"CLIP path not found at {clip_path}. Will load on-demand.")

        # UNET
        if useQ6:
            unet_path = os.environ.get("UNET_PATH", "wan2.1-t2v-14b-Q6_K.gguf")
        else:
            unet_path = os.environ.get("UNET_PATH", "wan2.1-t2v-14b-Q5_0.gguf")
        if Path(unet_path).exists():
            app.logger.info("Loading UNET into memory...")
            LOADED["unet"] = unet_loader.load_unet(unet_path)[0]
        else:
            app.logger.warning(f"UNET path not found at {unet_path}. Will load on-demand.")

        # VAE
        vae_path = os.environ.get("VAE_PATH", "wan_2.1_vae.safetensors")
        if Path(vae_path).exists():
            app.logger.info("Loading VAE into memory...")
            LOADED["vae"] = vae_loader.load_vae(vae_path)[0]
        else:
            app.logger.warning(f"VAE path not found at {vae_path}. Will load on-demand.")
    except Exception as e:
        app.logger.error("Model load failed: " + str(e))
        app.logger.error(traceback.format_exc())

# -------------------------
# WEBSOCKET BROADCAST
# -------------------------
def broadcast_ws(job_id, message):
    """
    Broadcast JSON message to all ws clients listening for job_id.
    message should be JSON-serializable.
    """
    try:
        conns = WS_CLIENTS.get(job_id, [])
        for ws in list(conns):
            try:
                ws.send(json.dumps(message))
            except Exception:
                # remove broken conns
                try:
                    conns.remove(ws)
                except Exception:
                    pass
    except Exception:
        pass

# -------------------------
# BACKGROUND WORKER & ROUTES
# -------------------------
def start_job(payload):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"status":"queued", "progress":0, "out":None, "logs":[], "last_log": "", "error": None}

    def worker():
        try:
            JOBS[job_id]["status"] = "running"
            JOBS[job_id]["progress"] = 0
            broadcast_ws(job_id, {"type":"log","text":"Job started."})
            # Call generate_video_sync with job id
            out_path = generate_video_sync(
                positive_prompt=payload.get("positive"),
                negative_prompt=payload.get("negative"),
                width=int(payload.get("width",832)),
                height=int(payload.get("height",480)),
                seed=int(payload.get("seed",0) or 0),
                steps=int(payload.get("steps",30)),
                cfg_scale=float(payload.get("cfg_scale",1.0)),
                sampler_name=payload.get("sampler_name","uni_pc"),
                scheduler=payload.get("scheduler","simple"),
                frames=int(payload.get("frames",33)),
                fps=int(payload.get("fps",16)),
                output_format=payload.get("output_format","mp4"),
                job_id=job_id,
                model_variant=payload.get("model")
            )
            JOBS[job_id]["out"] = out_path
            JOBS[job_id]["progress"] = 100
            JOBS[job_id]["status"] = "done"
            broadcast_ws(job_id, {"type":"done", "text":"Job finished."})
        except Exception as e:
            JOBS[job_id]["status"] = "error"
            JOBS[job_id]["error"] = str(e)
            JOBS[job_id]["logs"].append("Exception: " + str(e))
            broadcast_ws(job_id, {"type":"error", "text": str(e)})
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return job_id

@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(force=True)
    if not payload:
        return jsonify({"error":"invalid payload"}), 400
    job_id = start_job(payload)
    return jsonify({"id": job_id})

@app.route("/api/result/<job_id>", methods=["GET"])
def api_result(job_id):
    info = JOBS.get(job_id)
    if not info:
        return ("", 404)
    if info.get("status") != "done":
        return ("", 404)
    out = info.get("out")
    if not out or not os.path.exists(out):
        return ("", 404)
    # Return the file
    return send_file(out, as_attachment=False)

@app.route("/api/status/<job_id>", methods=["GET"])
def api_status(job_id):
    info = JOBS.get(job_id)
    if not info:
        return jsonify({"status":"notfound"}), 404
    return jsonify({
        "status": info.get("status"),
        "progress": info.get("progress"),
        "last_log": info.get("last_log"),
        "error": info.get("error")
    })

# WebSocket progress endpoint
@sock.route("/ws/progress/<job_id>")
def ws_progress(ws, job_id):
    # register ws
    conns = WS_CLIENTS.setdefault(job_id, [])
    conns.append(ws)
    try:
        # send initial state
        info = JOBS.get(job_id, {"status":"notfound", "progress":0})
        ws.send(json.dumps({"type":"progress", "progress": info.get("progress",0), "text": info.get("status","idle")}))
        # keep socket open until job done/error
        while True:
            # if job status changed, push updates (simple polling)
            info = JOBS.get(job_id)
            if info is None:
                ws.send(json.dumps({"type":"log", "text":"no such job"}))
                break
            ws.send(json.dumps({"type":"progress", "progress": info.get("progress",0), "text": info.get("status","")}))
            if info.get("status") in ("done","error"):
                if info.get("status") == "done":
                    ws.send(json.dumps({"type":"done"}))
                else:
                    ws.send(json.dumps({"type":"error", "text": info.get("error","unknown")}))
                break
            time.sleep(0.8)
    except Exception:
        pass
    finally:
        try:
            conns.remove(ws)
        except Exception:
            pass

# -------------------------
# STARTUP
# -------------------------
if __name__ == "__main__":
    # call model loading before starting server
    print("Loading models at startup (this may take a while)...")
    load_models_on_startup()
    print("Model loading complete — starting Flask.")
    # Run with gunicorn for production; for dev use this
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
