import sys
import json
import os
import time
import requests
from datetime import datetime

def generate_report(telemetry_file, session_name):
    # Signal that we are generating
    os.makedirs("data/ai_reports", exist_ok=True)
    status_file = "data/ai_reports/.generating"
    with open(status_file, "w") as f:
        f.write(session_name)
        
    try:
        with open(telemetry_file, "r") as f:
            events = json.load(f)
            
        if not events:
            return

        print(f"\n[AI Background Worker] Waking up Deep Semantic SLM for Final Report ({session_name})...")
        
        telemetry_lines = []
        for e in events:
            ts = e.get("ts", "")
            typ = e.get("type", "")
            d = e.get("data", {})
            info = []
            if "student" in d: info.append(f"Student: {d['student']}")
            if "conf" in d: info.append(f"Sim: {d['conf']}%")
            if "source" in d: info.append(f"Source: {d['source']}")
            if "action" in d: info.append(f"Action: {d['action']}")
            if "present" in d: info.append(f"Present: {d['present']}/{d.get('total','')}")
            telemetry_lines.append(f"[{ts}] {typ} | " + ", ".join(info))
        
        log_chunk = "\n".join(telemetry_lines)
        
        system_prompt = (
            "You are the AI Orchestrator for an advanced computer vision attendance system. "
            "A class session has just ended. Review the following telemetry log and write a concise, "
            "professional summary report. Include:\n"
            "1. Overall attendance summary (who was present based on the final heartbeat).\n"
            "2. Interesting timeline events (late arrivals, unknown visitors).\n"
            "3. System learning (who had their embedding caches 'expanded' or 'refined', indicating the system learned their face better today).\n"
            "Keep it structured and professional."
        )
        
        payload = {
            "model": "qwen2.5:1.5b",
            "prompt": f"{system_prompt}\n\nTELEMETRY:\n{log_chunk}\n\nREPORT:\n",
            "stream": False
        }

        start_t = time.time()
        print(f"  > Sending {len(events)} telemetry events to local Qwen2.5 model...")
        
        # Increase timeout to 300s to handle huge payloads
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        
        if response.status_code == 200:
            report_text = response.json().get("response", "")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/ai_reports/ai_report_{session_name}_{timestamp}.md"
            
            final_md = f"# AI Orchestrator Session Report\n**Session:** {session_name}\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            final_md += report_text
            
            with open(filename, "w") as f:
                f.write(final_md)
                
            print(f"  > [Success] Deep Semantic Report generated in {time.time()-start_t:.1f}s")
            print(f"  > Saved to: {filename}")
        else:
            print(f"[AI Background Worker] API error: {response.status_code}")
            
    except Exception as e:
        print(f"[AI Background Worker] Fatal error: {e}")
    finally:
        # Cleanup
        try: os.remove(telemetry_file)
        except: pass
        try: os.remove(status_file)
        except: pass

if __name__ == "__main__":
    if len(sys.argv) == 3:
        generate_report(sys.argv[1], sys.argv[2])
