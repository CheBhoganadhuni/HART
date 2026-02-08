# Phase 2b: MP4 Video Processing & Multi-Tab Safety

**Duration**: Phase 2a → Phase 2b  
**Status**: ✅ Completed  
**Date**: February 8, 2026

---

## 1. Overview

Phase 2b extends the web dashboard to support **MP4 video processing** alongside live sessions, with robust **multi-tab and multi-session safety mechanisms** to prevent concurrent process conflicts.

### Key Achievements
- ✅ MP4 video upload and processing via web UI
- ✅ Real-time progress tracking with frame counter
- ✅ Pause/Resume controls for MP4 processing
- ✅ Global system lock preventing concurrent sessions
- ✅ Full-page blocking overlay for multi-tab protection
- ✅ Pre-start race condition handling (409 status)
- ✅ Export path display for processed videos

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GLOBAL SYSTEM LOCK                          │
│       CV_PROCESS + PROCESS_TYPE = {"webcam" | "mp4" | None}     │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
      ┌──────────────┐               ┌──────────────┐
      │   /index     │               │ /process-mp4 │
      │  (Webcam)    │               │  (MP4 Page)  │
      └──────────────┘               └──────────────┘
              │                               │
              │    ✖ If other is running     │
              ├───────────────────────────────┤
              │                               │
              ▼                               ▼
    ┌──────────────────────────────────────────────────┐
    │                    views.py                       │
    │  - start_session() / start_mp4()                 │
    │  - Check PROCESS_TYPE before starting            │
    │  - Return 409 if busy                            │
    └──────────────────────────────────────────────────┘
              │
              ▼
    ┌──────────────────────────────────────────────────┐
    │                  main.py --web                    │
    │  - Reads commands.json (stop/pause/resume)       │
    │  - Writes mp4_progress.json (frame/total)        │
    │  - Outputs to live_stream.jpg                    │
    │  - Exports to data/exports/session.mp4           │
    └──────────────────────────────────────────────────┘
```

### Multi-Tab Safety Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  User opens Tab A → Starts MP4 processing                       │
│  User opens Tab B → Navigates to /index                         │
│                                                                 │
│  Tab B receives: is_busy=True, busy_type="MP4"                  │
│                     ↓                                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │         FULL-PAGE BLOCKING OVERLAY                       │   │
│  │  ⚠️ System Busy                                         │   │
│  │  A MP4 session is currently active.                     │   │
│  │  Please wait or stop it from the original tab.          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Tab A: Controls remain functional                              │
│  Tab B: Completely blocked (no interference)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `mp4.html` | MP4 processing page with upload, progress, and results |

### Modified Files
| File | Changes |
|------|---------|
| `views.py` | Added MP4 endpoints, pause_mp4, is_busy logic, 409 responses |
| `urls.py` | Added MP4 routes: upload, start, stop, pause, progress, results |
| `index.html` | Added busy overlay, systemBusy JS guards, 409 handling |
| `main.py` | Progress file output, pause/resume support |

---

## 4. Features Implemented

### 4.1 MP4 Video Processing
- **Drag & Drop Upload**: Modern upload zone with validation (MP4, <2GB)
- **Section Selection**: Existing section database for student matching
- **Session Naming**: Custom or auto-generated session identifier
- **Export Toggle**: Optional export of processed video

### 4.2 Progress Tracking
- **Progress Bar**: Visual percentage with frame counter
- **Frame Display**: "Frame 1234 / 5678" real-time updates
- **File-based IPC**: `mp4_progress.json` polled every second

### 4.3 Pause/Resume Control
- **Pause Button**: Toggles to Resume (yellow → green)
- **Command File**: Writes `{"pause": true}` or `{"resume": true}`
- **State Preservation**: Processing resumes at exact frame

### 4.4 Results Display
- **Attendance Count**: Present / Total / Unknowns
- **Export Path**: Shows `data/exports/session.mp4`
- **Attendee List**: Names with student IDs

### 4.5 Multi-Tab Protection

| Scenario | Behavior |
|----------|----------|
| Live running → Open MP4 tab | Full-page block overlay |
| MP4 running → Open Live tab | Full-page block overlay |
| Live running → Open Live tab | Block overlay (no interference) |
| MP4 running → Open MP4 tab | Same session, can control |
| 2 tabs open → Click start on both | Second tab: 409 → alert → reload |

### 4.6 Server-Side Safety
- **Global Lock**: `CV_PROCESS` + `PROCESS_TYPE` prevent concurrent sessions
- **409 Conflict**: Returns HTTP 409 if start called while busy
- **Type Tracking**: "webcam" vs "mp4" for appropriate messaging

---

## 5. Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Full-page overlay vs banner** | Complete interaction block, no accidental clicks |
| **Server-side is_busy** | Single source of truth, rendered at page load |
| **JS systemBusy guards** | Prevents beforeunload/stop interference |
| **409 + page reload** | Clean state reset for race conditions |
| **File path instead of URL** | Simpler, user knows exact file location |

---

## 6. API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/process-mp4` | GET | Render MP4 processing page |
| `/upload-mp4` | POST | Upload video file |
| `/start-mp4` | POST | Start MP4 processing |
| `/stop-mp4` | POST | Stop MP4 processing |
| `/pause-mp4` | POST | Toggle pause/resume |
| `/mp4-progress` | GET | Get current progress (JSON) |
| `/mp4-results` | GET | Get final results (JSON) |

---

## 7. Handled Edge Cases

1. **Open 2 tabs → Start on both**: Second gets 409, alert, reload
2. **Live running → Navigate to MP4**: Blocked with overlay
3. **MP4 running → Navigate to Live**: Blocked with overlay
4. **Close tab during session**: `beforeunload` guarded by systemBusy
5. **Refresh during processing**: Page reloads, shows processing view
6. **Browser back/forward**: Overlay appears if busy

---

## 8. Running Phase 2b

```bash
cd web_dashboard
python manage.py runserver

# Access dashboard: http://127.0.0.1:8000
# Access MP4 page: http://127.0.0.1:8000/process-mp4
```

---

## 9. Summary

Phase 2b successfully implements MP4 video processing with comprehensive multi-tab and multi-session safety:

- **Robust Locking**: Single global process prevents conflicts
- **User-Friendly Blocking**: Clear overlays explain system state
- **Race Condition Handling**: 409 status with graceful recovery
- **Feature Parity**: Pause/Resume mirrors live session controls

The architecture ensures that regardless of how users interact with multiple tabs or sessions, the system maintains data integrity and provides clear feedback.

---

## 10. Next Phases

### Phase 2c: Reports & Admin
- CSV/PDF export of attendance
- Session history with filtering
- Student management via web UI

### Phase 2d: External Cameras
- IP camera (CCTV) support
- Phone as webcam integration
- RTSP stream handling
