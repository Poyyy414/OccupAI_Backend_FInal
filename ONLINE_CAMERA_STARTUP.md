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

## 3. Publish the processed car video

Open a third PowerShell terminal and run:

```powershell
ffmpeg -fflags nobuffer -flags low_delay -i http://127.0.0.1:8001/stream -an -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -r 15 -g 15 -keyint_min 15 -sc_threshold 0 -bf 0 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/cars
```

Keep the terminal open. This publishes the YOLO-annotated car feed, rather
than opening the USB camera a second time.

## 4. Publish the processed motorcycle video

Open a fourth PowerShell terminal and run:

```powershell
ffmpeg -fflags nobuffer -flags low_delay -i http://127.0.0.1:8002/stream -an -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p -r 15 -g 15 -keyint_min 15 -sc_threshold 0 -bf 0 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/motorcycles
```

Keep the terminal open.

## 5. Check MediaMTX locally

Open these addresses in the laptop browser:

- Car: `http://127.0.0.1:8888/cars/`
- Motorcycle: `http://127.0.0.1:8888/motorcycles/`

Both videos should contain the computer-vision overlays. A short delay is
normal because the feeds use Low-Latency HLS.

## 6. Start the HTTPS tunnel

Open a fifth PowerShell terminal and run:

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

## 7. Update Render

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

After Render finishes deploying, sign in to the admin dashboard and driver
dashboard. Confirm that both live feeds appear. The Flutter app receives the
same addresses from the backend automatically.

## Normal startup checklist

- [ ] Both USB cameras are connected.
- [ ] MediaMTX is running.
- [ ] Both YOLO workers are running.
- [ ] Car FFmpeg publisher is running.
- [ ] Motorcycle FFmpeg publisher is running.
- [ ] Both local MediaMTX pages work.
- [ ] Cloudflare Tunnel is running.
- [ ] Both public HTTPS camera pages work.
- [ ] Render has the current `MEDIA_STREAM_BASE_URL`.
- [ ] Admin, driver web, and Flutter feeds work.

## How to stop everything safely

Press `Ctrl+C` once in each terminal:

1. Cloudflare Tunnel
2. Car FFmpeg
3. Motorcycle FFmpeg
4. Two-camera YOLO launcher
5. MediaMTX

Do not force-close the computer while model files or configuration files are
being edited.

## After a camera is disconnected

1. Stop both FFmpeg publishers with `Ctrl+C`.
2. Stop the YOLO launcher with `Ctrl+C`.
3. Reconnect both cameras.
4. Wait a few seconds for Windows to recognize them.
5. Start again from step 2.

MediaMTX can normally remain running. If the streams do not return, restart
MediaMTX too and follow the complete guide from step 1.

## Common problems

### Camera is busy or cannot be opened

Close Camera, OBS, Zoom, browser camera tests, and old detector or FFmpeg
processes. In this setup only the YOLO worker opens each USB camera. FFmpeg
reads the processed HTTP feed.

### FFmpeg says connection refused on port 8001 or 8002

The YOLO workers are not ready. Verify the local annotated feed in a browser,
then rerun the corresponding FFmpeg command.

### MediaMTX says port 8000 or another port is already in use

Another MediaMTX process may already be running. Use the existing process, or
stop it before starting another copy. The current configuration uses RTSP over
TCP to avoid the earlier RTSP UDP port conflict.

### Public address stopped working

The Quick Tunnel terminal was closed, the laptop lost internet access, or the
tunnel restarted. Run the Cloudflare command again, copy the new address, and
update `MEDIA_STREAM_BASE_URL` in Render.

### Video works but has several seconds of delay

Confirm the FFmpeg commands use `-preset ultrafast`, `-tune zerolatency`,
`-g 15`, `-keyint_min 15`, and `-bf 0`. Approximately one to three seconds on
the website and two to four seconds in the Flutter app can still be normal for
HLS. Sub-second internet playback generally requires WebRTC plus publicly
reachable ICE/TURN networking.

### Video has no parking boxes or detection labels

FFmpeg is probably reading the USB camera directly. Stop it and use the commands
in steps 3 and 4, which read the annotated YOLO feeds on ports 8001 and 8002.
