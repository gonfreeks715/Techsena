from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List
from pydantic import BaseModel
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import cm
import logging
import sys
import os
import io
import base64

# ── LOGGING SETUP ────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cargo-ai")

# ── MongoDB ───────────────────────────────────────────────────────────────────
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://sushantsgaikwad10_db_user:vSEoCg2TKYv4s46y@cluster0.cwed260.mongodb.net/cargoxai?retryWrites=true&w=majority"

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db                  = client["cargoxai"]
    collection          = db["devloper"]
    feedback_collection = db["feedback"]
    logger.info("✅ Connected to MongoDB")
    logger.info(f"📂 Collections: {db.list_collection_names()}")
except Exception as e:
    logger.error("❌ MongoDB connection failed: %s", str(e))
    raise e

# ── ML Model ──────────────────────────────────────────────────────────────────
sys.path.append('../ml_model')
from model import inspector

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="CargoVision AI API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── WebSocket Alert Manager ───────────────────────────────────────────────────
class AlertManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        logger.info("🔌 WebSocket connected")

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)
            logger.info("❌ WebSocket disconnected")

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.connections:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

alert_manager = AlertManager()


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":      "CargoVision AI is running",
        "version":     "3.0.0",
        "models":      inspector.yolo_names,
        "autoencoder": inspector.autoencoder is not None,
        "model5":      inspector.model5 is not None,
    }


# ── INSPECT ───────────────────────────────────────────────────────────────────
@app.post("/api/inspect")
async def inspect_cargo(
    file: UploadFile = File(...),
    declared_type: str = Form(default="unknown"),
    shipment_id:   str = Form(default="AUTO"),
):
    try:
        logger.info("🚀 API HIT — /api/inspect")

        # Step 1: Read file
        image_bytes = await file.read()
        logger.info(f"📁 File: {file.filename} | Size: {len(image_bytes)} bytes")

        # Step 2: Run ensemble model
        logger.info("🧠 Running ensemble model...")
        result = inspector.inspect(image_bytes, declared_type)

        if not result:
            raise Exception("Model returned empty result")

        # Step 3: Build DB record
        sid = (
            shipment_id
            if shipment_id != "AUTO"
            else f"SHP-{datetime.now().timestamp()}"
        )

        scan_record = {
            "id":              sid,
            "timestamp":       datetime.now().isoformat(),
            "filename":        file.filename,
            "declared_type":   declared_type,
            "risk_score":      result.get("risk_score"),
            "risk_level":      result.get("risk_level"),
            "total_objects":   result.get("total_objects"),
            "mismatches":      len(result.get("mismatches", [])),
            "models_used":     result.get("models_used", 1),
            "model_breakdown": result.get("model_breakdown", []),
        }

        logger.info(f"📦 Risk: {scan_record['risk_level']} | Score: {scan_record['risk_score']}")

        # Step 4: Save to MongoDB
        insert_result = collection.insert_one(scan_record)
        logger.info(f"✅ MongoDB inserted: {insert_result.inserted_id}")

        # Step 5: WebSocket alert for HIGH / MEDIUM
        if result.get("risk_level") in ["HIGH", "MEDIUM"]:
            emoji = "🚨" if result["risk_level"] == "HIGH" else "⚠️"
            await alert_manager.broadcast({
                "type":        "RISK_ALERT",
                "level":       result["risk_level"],
                "shipment_id": sid,
                "risk_score":  result["risk_score"],
                "message":     f"{emoji} {result['risk_level']} RISK detected in {sid}",
                "mismatches":  len(result.get("mismatches", [])),
                "timestamp":   scan_record["timestamp"],
            })

        return JSONResponse({"success": True, "shipment_id": sid, **result})

    except Exception as e:
        logger.error("❌ Inspect error: %s", str(e))
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── FEEDBACK ──────────────────────────────────────────────────────────────────
class FeedbackData(BaseModel):
    shipment_id: str
    rating:      int
    text:        str

@app.post("/api/feedback")
def submit_feedback(data: FeedbackData):
    try:
        feedback_collection.insert_one({
            "shipment_id": data.shipment_id,
            "rating":      data.rating,
            "text":        data.text,
            "timestamp":   datetime.now().isoformat(),
        })
        logger.info(f"📝 Feedback saved for {data.shipment_id}")
        return {"success": True}
    except Exception as e:
        logger.error("❌ Feedback error: %s", str(e))
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── HISTORY ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_history():
    try:
        data = list(
            collection.find({}, {"_id": 0})
            .sort("timestamp", -1)
            .limit(20)
        )
        logger.info(f"📜 History: {len(data)} records")
        return {"scans": data}
    except Exception as e:
        logger.error("❌ History error: %s", str(e))
        return {"scans": []}


# ── STATS ─────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def get_stats():
    try:
        pipeline = [
            {"$group": {"_id": "$declared_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        categories_db = list(collection.aggregate(pipeline))
        categories = [{"name": str(c["_id"]).title() if c["_id"] else "Unknown", "count": c["count"]} for c in categories_db]
        
        return {
            "total":       collection.count_documents({}),
            "high_risk":   collection.count_documents({"risk_level": "HIGH"}),
            "medium_risk": collection.count_documents({"risk_level": "MEDIUM"}),
            "low_risk":    collection.count_documents({"risk_level": "LOW"}),
            "categories":  categories,
        }
    except Exception as e:
        logger.error("❌ Stats error: %s", str(e))
        return {}


# ── TEST DB ───────────────────────────────────────────────────────────────────
@app.get("/test-db")
def test_db():
    try:
        collection.insert_one({"test": "working", "time": datetime.now().isoformat()})
        return {"message": "✅ MongoDB working"}
    except Exception as e:
        logger.error("❌ DB test failed: %s", str(e))
        return {"message": "❌ DB failed"}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await alert_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        alert_manager.disconnect(websocket)


# ── FULL PDF REPORT ───────────────────────────────────────────────────────────
@app.post("/api/generate-report")
async def generate_report(request: dict):
    """
    Full professional dark-themed PDF report.
    Called from frontend Download Report button.
    Accepts the full result object from /api/inspect.
    """
    try:
        scan_id    = request.get("shipment_id",    "SHP-0001")
        risk_score = request.get("risk_score",     0)
        risk_level = request.get("risk_level",     "LOW")
        declared   = request.get("declared_type",  "unknown")
        detections = request.get("detections",     [])
        mismatches = request.get("mismatches",     [])
        total_objs = request.get("total_objects",  0)
        breakdown  = request.get("model_breakdown",[])
        img_b64    = request.get("annotated_image", None)
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        os.makedirs("reports", exist_ok=True)
        pdf_path = f"reports/report_{scan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            rightMargin=1.5*cm, leftMargin=1.5*cm,
            topMargin=1.5*cm,   bottomMargin=1.5*cm,
        )

        styles = getSampleStyleSheet()
        normal = styles["Normal"]
        elems  = []

        # ── colour helpers ────────────────────────────────────────────────
        risk_hex = (
            "#ef4444" if risk_level == "HIGH"   else
            "#f59e0b" if risk_level == "MEDIUM" else
            "#10b981"
        )
        dark_bg  = colors.HexColor("#0f1628")
        accent   = colors.HexColor("#1a73e8")
        card_bg  = colors.HexColor("#131c30")
        border_c = colors.HexColor("#1e2d4a")

        def P(html):
            return Paragraph(html, normal)

        def tbl(data, widths, cmds):
            t = Table(data, colWidths=widths)
            t.setStyle(TableStyle(cmds))
            return t

        BASE_CMDS = [
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
        ]

        # ── HEADER ───────────────────────────────────────────────────────
        elems += [
            tbl([[
                P("<font color='#1a73e8' size='18'><b>CARGOVISION AI</b></font><br/>"
                  "<font color='#5a6a85' size='9'>AI-ASSISTED CARGO INSPECTION — ENSEMBLE REPORT</font>"),
                P(f"<font color='#5a6a85' size='8'>GENERATED<br/>"
                  f"<font color='#e8edf5'>{timestamp}</font></font>"),
            ]], [12*cm, 6*cm],
            BASE_CMDS + [("BACKGROUND",(0,0),(-1,-1), dark_bg)]),
            Spacer(1, 0.4*cm),
        ]

        # ── RISK BANNER ──────────────────────────────────────────────────
        elems += [
            tbl([[P(
                f"<font color='white' size='13'><b>"
                f"{risk_level} RISK — Score: {risk_score}/100</b></font><br/>"
                f"<font color='white' size='9'>"
                f"Shipment: {scan_id}  |  Declared: {declared}  "
                f"|  Objects: {total_objs}  |  Mismatches: {len(mismatches)}</font>"
            )]], [18*cm],
            BASE_CMDS + [("BACKGROUND",(0,0),(-1,-1), colors.HexColor(risk_hex))]),
            Spacer(1, 0.4*cm),
        ]

        # ── SCAN INFO ────────────────────────────────────────────────────
        def info_row(lbl, val, vc="#e8edf5"):
            return [
                P(f"<font color='#5a6a85' size='9'>{lbl}</font>"),
                P(f"<font color='{vc}' size='10'><b>{val}</b></font>"),
            ]

        elems += [
            tbl([
                info_row("SHIPMENT ID",    scan_id),
                info_row("RISK LEVEL",     risk_level, risk_hex),
                info_row("ENSEMBLE SCORE", f"{risk_score} / 100"),
                info_row("DECLARED TYPE",  declared.upper()),
                info_row("OBJECTS FOUND",  str(total_objs)),
                info_row("MISMATCHES",     str(len(mismatches))),
                info_row("GENERATED AT",   timestamp),
            ], [5*cm, 13*cm],
            BASE_CMDS + [
                ("ROWBACKGROUNDS",(0,0),(-1,-1),[card_bg, colors.HexColor("#0f1628")]),
                ("LINEBELOW",(0,0),(-1,-2), 0.5, border_c),
            ]),
            Spacer(1, 0.4*cm),
        ]

        # ── MODEL BREAKDOWN ──────────────────────────────────────────────
        if breakdown:
            elems.append(P("<font color='#5a6a85' size='9'>ENSEMBLE MODEL BREAKDOWN</font>"))
            elems.append(Spacer(1, 0.2*cm))

            bd_rows = [[
                P("<font color='#5a6a85' size='8'><b>MODEL</b></font>"),
                P("<font color='#5a6a85' size='8'><b>SCORE</b></font>"),
                P("<font color='#5a6a85' size='8'><b>LEVEL</b></font>"),
            ]]
            for b in breakdown:
                sc  = b.get("score", 0)
                col = "#ef4444" if sc >= 70 else "#f59e0b" if sc >= 40 else "#10b981"
                lvl = "HIGH" if sc >= 70 else "MEDIUM" if sc >= 40 else "LOW"
                bd_rows.append([
                    P(f"<font color='#e8edf5' size='10'>{b.get('model','')}</font>"),
                    P(f"<font color='{col}' size='10'><b>{sc:.1f}</b></font>"),
                    P(f"<font color='{col}' size='10'>{lvl}</font>"),
                ])
            # Ensemble average row
            bd_rows.append([
                P("<font color='#1a73e8' size='10'><b>ENSEMBLE AVERAGE</b></font>"),
                P(f"<font color='{risk_hex}' size='10'><b>{risk_score}</b></font>"),
                P(f"<font color='{risk_hex}' size='10'><b>{risk_level}</b></font>"),
            ])

            elems += [
                tbl(bd_rows, [7*cm, 5*cm, 6*cm],
                BASE_CMDS + [
                    ("BACKGROUND",    (0,0), (-1,0),  colors.HexColor("#1e2d4a")),
                    ("ROWBACKGROUNDS",(0,1), (-1,-2), [card_bg, colors.HexColor("#0f1628")]),
                    ("BACKGROUND",    (0,-1),(-1,-1), colors.HexColor("#0d1f3a")),
                    ("LINEBELOW",     (0,0), (-1,-1), 0.5, border_c),
                ]),
                Spacer(1, 0.4*cm),
            ]

        # ── DETECTIONS ───────────────────────────────────────────────────
        elems.append(P("<font color='#5a6a85' size='9'>DETECTED OBJECTS</font>"))
        elems.append(Spacer(1, 0.2*cm))

        THREAT_WORDS = ["gun","knife","weapon","explosive","drug",
                        "firearm","pistol","rifle","blade","grenade"]

        det_rows = [[
            P("<font color='#5a6a85' size='8'><b>OBJECT</b></font>"),
            P("<font color='#5a6a85' size='8'><b>CONFIDENCE</b></font>"),
            P("<font color='#5a6a85' size='8'><b>MODELS</b></font>"),
            P("<font color='#5a6a85' size='8'><b>THREAT</b></font>"),
        ]]
        for d in detections:
            thr = any(t in d.get("label","").lower() for t in THREAT_WORDS)
            tc  = "#ef4444" if thr else "#10b981"
            src = d.get("sources", [d.get("source","?")])
            mc  = len(src) if isinstance(src, list) else 1
            det_rows.append([
                P(f"<font color='#e8edf5' size='10'>{d.get('label','')}</font>"),
                P(f"<font color='#06b6d4' size='10'>{d.get('confidence',0)*100:.1f}%</font>"),
                P(f"<font color='#5a6a85' size='10'>{mc} model(s)</font>"),
                P(f"<font color='{tc}' size='10'>{'YES' if thr else 'No'}</font>"),
            ])
        if len(det_rows) == 1:
            det_rows.append([
                P("<font color='#5a6a85' size='9'>No objects detected</font>"),
                P(""), P(""), P(""),
            ])

        elems += [
            tbl(det_rows, [6*cm, 4*cm, 4*cm, 4*cm],
            BASE_CMDS + [
                ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#1e2d4a")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1), [card_bg, colors.HexColor("#0f1628")]),
                ("LINEBELOW",     (0,0),(-1,-1), 0.5, border_c),
            ]),
            Spacer(1, 0.4*cm),
        ]

        # ── MISDECLARATION ───────────────────────────────────────────────
        if mismatches:
            elems.append(P("<font color='#ef4444' size='9'>MISDECLARATION ALERT</font>"))
            elems.append(Spacer(1, 0.2*cm))
            for m in mismatches:
                elems += [
                    tbl([[P(
                        f"<font color='#ef4444' size='10'><b>MISMATCH DETECTED</b></font><br/>"
                        f"<font color='#e8edf5' size='9'>"
                        f"Declared: <b>{m.get('declared','').upper()}</b>  →  "
                        f"Detected: <b>{m.get('detected','')}</b>  |  "
                        f"Severity: <b>{m.get('severity','HIGH')}</b>"
                        f"</font>"
                    )]], [18*cm],
                    BASE_CMDS + [
                        ("BACKGROUND", (0,0),(-1,-1), colors.HexColor("#2d0a0a")),
                        ("LINELEFT",   (0,0),(0,-1),  4, colors.HexColor("#ef4444")),
                    ]),
                    Spacer(1, 0.2*cm),
                ]

        # ── ANNOTATED IMAGE ──────────────────────────────────────────────
        if img_b64:
            try:
                from reportlab.platypus import Image as RLImage
                elems += [
                    Spacer(1, 0.3*cm),
                    P("<font color='#5a6a85' size='9'>ANNOTATED SCAN</font>"),
                    Spacer(1, 0.2*cm),
                    RLImage(
                        io.BytesIO(base64.b64decode(img_b64)),
                        width=16*cm, height=10*cm,
                    ),
                ]
            except Exception as img_err:
                logger.warning(f"⚠️ Could not embed image: {img_err}")

        # ── FOOTER ───────────────────────────────────────────────────────
        elems += [
            Spacer(1, 0.5*cm),
            tbl([[P(
                "<font color='#5a6a85' size='8'>"
                "Generated by CargoVision AI Ensemble System. "
                "For official use only. Verify all flagged items with a certified customs officer. "
                "© 2026 TechSena — CargoVision AI"
                "</font>"
            )]], [18*cm],
            BASE_CMDS + [
                ("BACKGROUND", (0,0),(-1,-1), dark_bg),
                ("LINEABOVE",  (0,0),(-1,-1), 0.5, accent),
            ]),
        ]

        doc.build(elems)
        logger.info(f"📄 PDF generated: {pdf_path}")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"CargoVision_Report_{scan_id}.pdf",
        )

    except Exception as e:
        logger.error("❌ PDF generation error: %s", str(e))
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ── SIMPLE PDF REPORT (original kept for compatibility) ───────────────────────
@app.post("/api/report")
async def simple_report():
    """Original simple report endpoint — kept for backward compatibility."""
    try:
        buffer = io.BytesIO()
        doc    = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        content = [
            Paragraph("Cargo Inspection Report", styles["Title"]),
            Spacer(1, 20),
            Paragraph("Status: Generated Successfully", styles["Normal"]),
            Spacer(1, 10),
            Paragraph("System: CargoVision AI", styles["Normal"]),
        ]
        doc.build(content)
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=report.pdf"},
        )
    except Exception as e:
        return {"error": str(e)}


# Run with: uvicorn main:app --reload --port 8000