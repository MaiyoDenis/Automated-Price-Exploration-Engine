# Project APEX — Automated Price EXploration

A research-first quantitative trading platform for Deriv with autonomous multi-strategy routing, risk management, and real-time dashboard analytics.

---

## 🚀 Quick Start (Local)

Run the application locally:
```bash
PYTHONPATH=. .venv/bin/python main.py
```
Open Dashboard in Browser:
```text
 
```

---

## 🌐 Remote & Mobile UI Access

Access your live trading dashboard from **any Laptop, Phone, or Tablet**:

- **Web Dashboard URL**: [https://assetmindsolution.com/apex/](https://assetmindsolution.com/apex/)
- **Direct VPS IP**: `http://65.21.55.114:8080`

---

## ⚙️ VPS Management Commands

The system runs as a managed `systemd` background service on your VPS (`65.21.55.114`).

| Action | Command |
| :--- | :--- |
| **Check Live Status** | `ssh root@65.21.55.114 "systemctl status project-apex"` |
| **View Live Logs** | `ssh root@65.21.55.114 "journalctl -u project-apex -f"` |
| **Restart Service** | `ssh root@65.21.55.114 "systemctl restart project-apex"` |
| **Stop Service** | `ssh root@65.21.55.114 "systemctl stop project-apex"` |
| **Start Service** | `ssh root@65.21.55.114 "systemctl start project-apex"` |

---

## 📊 Features & Architecture

- **Multi-Strategy Engine**: Holy Grail, Differs Adaptive V2, VWAP Scalper, MetaRegime Router, XGBoost ML.
- **Risk Protection**: Circuit breaker loss velocity, streak limits, daily max loss, drawdowns.
- **Self-Healing SQLite**: WAL mode with automatic corruption detection and instant background recovery.
- **Resource Guardrails**: Throttled to `Nice=10`, `CPUQuota=50%`, `MemoryMax=500M` to protect co-hosted VPS applications.

