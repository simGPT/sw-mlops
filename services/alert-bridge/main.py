import os
import httpx
from fastapi import FastAPI, Request

app = FastAPI(title="Alert Bridge")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO_OWNER = os.getenv("GITHUB_REPO_OWNER")
GITHUB_REPO_NAME = os.getenv("GITHUB_REPO_NAME")

# Alertmanager에서 알림을 받아서 GitHub Actions로 전달하는 역할을 하는 api 엔드포인트(Alertmanager가 요청을 보냄)
@app.post("/alert")
async def receive_alert(request: Request):
    payload = await request.json()

    # alertmanager에서 받은 알림 중에서 firing 상태인 알림만 필터링
    firing_alerts = [a for a in payload.get("alerts", []) if a["status"] == "firing"]
    if not firing_alerts:
        return {"message": "no firing alerts"}

    alert_names = [a["labels"].get("alertname", "") for a in firing_alerts]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/dispatches",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
            },
            json={
                "event_type": "model-retrain",
                "client_payload": {"alerts": alert_names},
            },
        )

    return {"status": response.status_code, "alerts": alert_names}
