# OccupAI Online Camera Startup Guide

Use this guide whenever the computer or cameras have been turned off.

## Before starting

1. Connect both USB cameras.
2. Connect the laptop to the internet.
3. Close applications that may be using the cameras, such as the Windows
   Camera app, OBS, Zoom, or a browser camera test.
4. Use a separate PowerShell terminal for every numbered process below.

Start the processes in this exact order.

## 1. Start MediaMTX

Open PowerShell and run:

```powershell
cd "$env:USERPROFILE\Downloads\mediamtx_v1.21.3_windows_amd64"
.\mediamtx.exe
```

Keep this terminal open. A successful startup should show listeners including
RTSP on port `8554` and HLS on port `8888`.

The active `mediamtx.yml` should contain:

```yaml
hlsVariant: lowLatency
hlsAlwaysRemux: true
hlsSegmentDuration: 1s
hlsPartDuration: 200ms
```

Keeping the HLS muxer warm prevents a new viewer from seeing a black frame while
MediaMTX waits to create the first playlist and segment.

## 2. Start the computer-vision workers

Open a second PowerShell terminal and run:

```powershell
cd "C:\Users\arias\Documents\OccupAI_Backend_FInal"
.\start_two_cameras.ps1
```

Keep this terminal open. It starts and supervises:

- Car detector and annotated feed: `http://127.0.0.1:8001/stream`
- Motorcycle detector and annotated feed: `http://127.0.0.1:8002/stream`

Open both addresses in the laptop browser before continuing. Confirm that both
feeds show the correct camera, parking boxes, status colors, and detection
information.

The launcher now starts an internal FFmpeg publisher for each detector. It
publishes fresh camera frames at 20 FPS with the latest CV overlay and a
controlled bitrate. YOLO updates the overlay independently, so inference does
not pause the video.
Do not run the old manual FFmpeg commands because two publishers cannot use the
same `cars` or `motorcycles` MediaMTX path.

## 3. Check MediaMTX locally

Open these addresses in the laptop browser:

- Car: `http://127.0.0.1:8888/cars/`
- Motorcycle: `http://127.0.0.1:8888/motorcycles/`

Both videos should contain the computer-vision overlays. A short delay is
normal because the feeds use Low-Latency HLS.

## 4. Start the HTTPS tunnel

Open a third PowerShell terminal and run:

```powershell
cloudflared tunnel --url http://127.0.0.1:8888
```

Keep this terminal open. Copy the generated address that looks like:

```text
https://random-words.trycloudflare.com
```

Test the new address in a browser:

```text
https://random-words.trycloudflare.com/cars/
https://random-words.trycloudflare.com/motorcycles/
```

## 5. Update Render

In the Render dashboard:

1. Open the OccupAI backend service.
2. Open **Environment**.
3. Set `MEDIA_STREAM_BASE_URL` to the new Cloudflare address.
4. Do not add a trailing `/`.
5. Save the environment change and allow Render to redeploy.

Example:

```text
MEDIA_STREAM_BASE_URL=https://random-words.trycloudflare.com
```

After Render finishes deploying:

- The admin dashboard defaults to **Local CV**. Use **Open local car CV** and
  **Open local motorcycle CV** on the camera laptop for the lowest-delay view.
  Occupancy, reservations, and payments still update through Render.
- Use **Cloud preview** only when the admin is away from the camera laptop.
- The driver dashboard and Flutter app always use the public cloud feeds. They
  receive the addresses from the backend automatically.

The local feed opens in its own browser tab because an HTTPS Render page cannot
safely embed an HTTP localhost stream. `127.0.0.1` refers to the device opening
the page, so Local CV must be used on the laptop running the two camera workers.

## Normal startup checklist

- [ ] Both USB cameras are connected.
- [ ] MediaMTX is running.
- [ ] Both YOLO workers are running.
- [ ] Each YOLO terminal reports its MediaMTX publisher is connected.
- [ ] Both local MediaMTX pages work.
- [ ] Cloudflare Tunnel is running.
- [ ] Both public HTTPS camera pages work.
- [ ] Render has the current `MEDIA_STREAM_BASE_URL`.
- [ ] Admin Local CV buttons open both direct localhost feeds.
- [ ] Admin Cloud preview works as a fallback.
- [ ] Driver web and Flutter cloud feeds work.

## How to stop everything safely

Press `Ctrl+C` once in each terminal:

1. Cloudflare Tunnel
2. Two-camera YOLO launcher
3. MediaMTX

Do not force-close the computer while model files or configuration files are
being edited.

## After a camera is disconnected

1. Stop the YOLO launcher with `Ctrl+C`.
2. Reconnect both cameras.
3. Wait a few seconds for Windows to recognize them.
4. Start again from step 2.

MediaMTX can normally remain running. If the streams do not return, restart
MediaMTX too and follow the complete guide from step 1.

## Common problems

### Camera is busy or cannot be opened

Close Camera, OBS, Zoom, browser camera tests, and old manual FFmpeg processes.
In this setup only the YOLO worker opens each USB camera and manages its own
background FFmpeg publisher.

### Detector repeatedly says the media publisher disconnected

Start MediaMTX before the YOLO launcher. Confirm RTSP port `8554` is available,
then restart the YOLO launcher. Also confirm `ffmpeg` works from PowerShell.

### MediaMTX says port 8000 or another port is already in use

Another MediaMTX process may already be running. Use the existing process, or
stop it before starting another copy. The current configuration uses RTSP over
TCP to avoid the earlier RTSP UDP port conflict.

### Public address stopped working

The Quick Tunnel terminal was closed, the laptop lost internet access, or the
tunnel restarted. Run the Cloudflare command again, copy the new address, and
update `MEDIA_STREAM_BASE_URL` in Render.

### Video works but has several seconds of delay

Confirm there are no old manual FFmpeg processes publishing to the same paths.
The integrated publisher uses a steady 20 FPS, one-second keyframes, no B-frames,
and a controlled 700 kbps stream. Approximately one to three seconds on the
website and two to four seconds in the Flutter app can still be normal for HLS.
Sub-second internet playback generally requires WebRTC plus publicly reachable
ICE/TURN networking.

### Video has no parking boxes or detection labels

Pull the latest detector code and restart `start_two_cameras.ps1`. The integrated
publisher sends the annotated detector frame, including green available, yellow
reserved, and red occupied parking boxes.
