import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.mime.text import MIMEText
from fastapi import FastAPI, Request

app = FastAPI(title="Alert Bridge")

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
ALERT_MAIL_FROM = os.getenv("ALERT_MAIL_FROM")
ALERT_MAIL_TO = os.getenv("ALERT_MAIL_TO")

# 알림 내용을 이메일로 작성하는 함수
def build_email_body(alerts: list) -> str:
    lines = [f"[sw-mlops 알림] {len(alerts)}개의 이상 감지 - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}\n"]
    for a in alerts:
        labels = a.get("labels", {})
        annotations = a.get("annotations", {})
        lines.append(f"• Alert   : {labels.get('alertname', '-')}")
        lines.append(f"  Severity: {labels.get('severity', '-')}")
        lines.append(f"  Summary : {annotations.get('summary', '-')}")
        lines.append(f"  Detail  : {annotations.get('description', '-')}\n")
    return "\n".join(lines)

# 이메일 전송하는 함수
def send_email(subject: str, body: str):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = ALERT_MAIL_FROM
    msg["To"] = ALERT_MAIL_TO

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls(context=ssl.create_default_context())
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(ALERT_MAIL_FROM, ALERT_MAIL_TO, msg.as_string())


# 알림 받아서 이메일 전송하는 api
@app.post("/alert")
async def receive_alert(request: Request):
    payload = await request.json()

    firing_alerts = [a for a in payload.get("alerts", []) if a["status"] == "firing"]
    if not firing_alerts:
        return {"message": "발생중인 알림이 없습니다."}

    alert_names = [a["labels"].get("alertname", "") for a in firing_alerts]
    subject = f"[sw-mlops 알림] {', '.join(alert_names)}"
    body = build_email_body(firing_alerts) # 알림 내용을 이메일 body로 작성

    send_email(subject, body) # 이메일 전송

    return {"message": "이메일이 전송되었습니다.", "alerts": alert_names}
