"""Flask backend.  Run: python app.py  ->  http://127.0.0.1:5000"""
import base64, io, math
import cv2, numpy as np, torch
from flask import Flask, jsonify, request
from model import SiameseNet
from preprocess import preprocess

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ck = torch.load("models/siamese.pt", map_location="cpu")
model = SiameseNet(); model.load_state_dict(ck["state"]); model.eval()
THR = ck["threshold"]


def b64(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/health")
def health():
    return jsonify(status="ok", threshold=THR)


@app.post("/api/verify")
def verify():
    try:
        ref, qst = request.files["reference"].read(), request.files["questioned"].read()
        p1, p2 = preprocess(ref), preprocess(qst)
    except Exception as e:
        return jsonify(error=f"Invalid image: {e}"), 400
    t = lambda p: torch.from_numpy(p).float().div(255)[None, None]
    with torch.no_grad():
        d = model(t(p1), t(p2)).item()
    conf = 1 / (1 + math.exp(8 * (d - THR) / THR))          # 50% exactly at the threshold
    return jsonify(distance=round(d, 4), threshold=round(THR, 4), similarity=round(conf * 100, 1),
                   genuine=bool(d <= THR), ref_processed=b64(p1), qst_processed=b64(p2))


if __name__ == "__main__":
    app.run(debug=False)
