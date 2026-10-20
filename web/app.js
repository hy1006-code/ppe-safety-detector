'use strict';
const el = Object.fromEntries(['camera','output','stage','emptyState','startButton','stopButton','saveButton','demoButton','confidence','confidenceValue','modelLabel','readyDot','liveDot','modeLabel','resolution','frameStatus','error','targetCount','latency','fps','detectionList'].map(id => [id, document.getElementById(id)]));
const context = el.output.getContext('2d');
let stream = null, active = false, generation = 0, timer = null, abort = null;
let snapshot = null, modelLabel = '', ready = false, pendingCamera = false;
const colors = ['#40d69b','#59cae2','#ffac59','#ffac59','#ffac59','#80bdff','#fae572','#40d69b','#bda7ff','#bda7ff'];
function error(message) { el.error.textContent = message; el.error.hidden = !message; }
function stop(message = '已停止检测，保留最后画面') {
  active = false; pendingCamera = false; generation++; clearTimeout(timer); timer = null;
  if (abort) { abort.abort(); abort = null; }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  el.camera.srcObject = null; el.startButton.disabled = !ready; el.stopButton.disabled = true;
  el.demoButton.disabled = !ready || el.demoButton.dataset.available !== 'true';
  el.liveDot.classList.remove('active'); el.fps.innerHTML = '—<small> 帧/秒</small>';
  if (snapshot?.mode === 'camera') el.modeLabel.textContent = '摄像头已停止';
  if (message) el.frameStatus.textContent = message;
}
function draw(frame, result) {
  const width = frame.width, height = frame.height;
  el.output.width = width; el.output.height = height; el.stage.style.aspectRatio = `${width}/${height}`;
  context.drawImage(frame, 0, 0); const font = Math.max(14, Math.round(width / 44));
  context.font = `600 ${font}px -apple-system, "PingFang SC", sans-serif`;
  context.lineWidth = Math.max(2, width / 300); context.textBaseline = 'top';
  const labelAreas = [];
  for (const box of result.boxes) {
    const [x1, y1, x2, y2] = box.xyxy; const x = x1 * width, y = y1 * height;
    const color = colors[box.class_id]; context.strokeStyle = color;
    context.strokeRect(x, y, (x2 - x1) * width, (y2 - y1) * height);
    const text = `${box.label} ${(box.confidence * 100).toFixed(0)}%`;
    const tw = context.measureText(text).width + 10, th = font + 8;
    const tx = Math.max(0, Math.min(x, width - tw)); let ty = Math.max(0, y - th), placed = false;
    for (let attempt = 0; attempt < 16 && ty + th <= height; attempt++, ty += th + 2) {
      if (!labelAreas.some(area => tx < area.x + area.w && tx + tw > area.x && ty < area.y + area.h && ty + th > area.y)) { placed = true; break; }
    }
    if (!placed) continue;
    labelAreas.push({x:tx, y:ty, w:tw, h:th});
    context.fillStyle = color; context.fillRect(tx, ty, tw, th);
    context.fillStyle = '#102330'; context.fillText(text, tx + 5, ty + 4);
  }
  el.emptyState.hidden = true; el.saveButton.disabled = false;
  el.resolution.textContent = `${width} × ${height}`; el.targetCount.textContent = `${result.boxes.length} 个`;
  el.latency.innerHTML = `${Math.round(result.inference_ms)}<small> ms</small>`;
  el.detectionList.replaceChildren();
  if (!result.boxes.length) {
    const p = document.createElement('p'); p.className = 'list-empty'; p.textContent = '当前阈值下未检测到目标'; el.detectionList.append(p);
  }
  for (const box of [...result.boxes].sort((a, b) => b.confidence - a.confidence).slice(0, 25)) {
    const row = document.createElement('div'); row.className = 'detection-row';
    const label = document.createElement('span'); const dot = document.createElement('i'); dot.className = 'label-dot'; dot.style.background = colors[box.class_id];
    label.append(dot, document.createTextNode(box.label)); const score = document.createElement('strong'); score.textContent = `${(box.confidence * 100).toFixed(1)}%`;
    row.append(label, score); el.detectionList.append(row);
  }
}
function blobFrom(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob(blob => blob ? resolve(blob) : reject(new Error('无法读取摄像头画面')), 'image/jpeg', 0.88));
}
async function predict(frame, token, mode) {
  const blob = await blobFrom(frame); if (token !== generation) return;
  const controller = new AbortController(); abort = controller;
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/api/predict?conf=${encodeURIComponent(el.confidence.value)}`, {method:'POST', headers:{'Content-Type':'image/jpeg'}, body:blob, signal:controller.signal});
    const data = await response.json(); if (!response.ok) throw new Error(data.error || '预测失败');
    if (token !== generation) return;
    draw(frame, data); snapshot = {mode, takenAt:new Date(), confidence:el.confidence.value, count:data.boxes.length};
    el.frameStatus.textContent = mode === 'camera' ? '正在实时检测 · 画面仅在本机处理' : '示例图片测试完成 · 非摄像头实拍';
    return data;
  } finally { clearTimeout(timeout); if (abort === controller) abort = null; }
}
async function loop(token) {
  if (!active || token !== generation) return;
  const started = performance.now();
  try {
    const video = el.camera; if (!video.videoWidth) throw new Error('摄像头尚未输出画面，请重新开启');
    const frame = document.createElement('canvas'); frame.width = Math.min(960, video.videoWidth);
    frame.height = Math.round(frame.width * video.videoHeight / video.videoWidth); frame.getContext('2d').drawImage(video, 0, 0, frame.width, frame.height);
    await predict(frame, token, 'camera');
    if (active && token === generation) {
      const elapsed = performance.now() - started;
      el.fps.innerHTML = `${(1000 / Math.max(150, elapsed)).toFixed(1)}<small> 帧/秒</small>`;
      timer = setTimeout(() => loop(token), Math.max(0, 150 - elapsed));
    }
  } catch (e) {
    if (token !== generation) return;
    stop('检测已停止'); error(e.name === 'AbortError' ? '模型响应超时，请确认本地服务仍在运行，再重试。' : e.message);
  }
}
async function start() {
  if (active || pendingCamera) return;
  if (!navigator.mediaDevices?.getUserMedia) return error('当前浏览器不支持摄像头访问。请用 Chrome 或 Safari 打开 http://127.0.0.1:8765 。');
  stop('正在请求摄像头权限…'); error(''); const token = generation; pendingCamera = true;
  el.startButton.disabled = true; el.stopButton.disabled = false; el.demoButton.disabled = true;
  try {
    const media = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:960}, height:{ideal:720}, facingMode:'user'}, audio:false});
    if (token !== generation) { media.getTracks().forEach(track => track.stop()); return; }
    stream = media; el.camera.srcObject = media; await el.camera.play();
    if (token !== generation) return;
    pendingCamera = false; active = true; el.modeLabel.textContent = '摄像头实时检测'; el.liveDot.classList.add('active');
    media.getVideoTracks()[0].addEventListener('ended', () => { if (active) { stop('摄像头连接已结束'); error('摄像头已断开，请重新开启。'); } }, {once:true});
    loop(token);
  } catch (e) {
    if (token !== generation) return;
    stop('摄像头未开启');
    const messages = {NotAllowedError:'摄像头权限未开启。请允许此网页使用摄像头；若仍无法打开，请在系统设置 → 隐私与安全性 → 摄像头中允许当前浏览器。', NotFoundError:'没有找到摄像头，请连接摄像头后重试。', NotReadableError:'摄像头正被其他应用占用或暂不可用，请关闭占用它的应用后重试。'};
    error(messages[e.name] || `无法打开摄像头：${e.message}`);
  }
}
async function demo() {
  stop('正在读取示例图片…'); error(''); const token = generation; el.demoButton.disabled = true; el.startButton.disabled = true;
  let bitmap;
  try {
    const response = await fetch('/api/demo'); if (!response.ok) throw new Error('本机没有示例图片');
    bitmap = await createImageBitmap(await response.blob()); if (token !== generation) return;
    const frame = document.createElement('canvas'); frame.width = bitmap.width; frame.height = bitmap.height; frame.getContext('2d').drawImage(bitmap, 0, 0);
    el.modeLabel.textContent = '示例图片 · 非摄像头实拍'; await predict(frame, token, 'demo');
  } catch (e) { if (token === generation) error(e.message); }
  finally { bitmap?.close(); if (token === generation) { el.demoButton.disabled = false; el.startButton.disabled = !ready; } }
}
function save() {
  if (!snapshot) return;
  const canvas = document.createElement('canvas'); canvas.width = el.output.width; canvas.height = el.output.height + 106;
  const ctx = canvas.getContext('2d'); ctx.fillStyle = '#122432'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#ffffff'; ctx.font = 'bold 21px -apple-system, "PingFang SC", sans-serif'; ctx.fillText('施工现场安全装备实时检测', 18, 32);
  ctx.font = '13px -apple-system, "PingFang SC", sans-serif'; ctx.fillStyle = '#b4c7d5';
  ctx.fillText(`${modelLabel}  |  ${snapshot.mode === 'camera' ? '摄像头实拍' : '示例图片测试（非实拍）'}`, 18, 57);
  ctx.drawImage(el.output, 0, 70);
  ctx.fillText(`${snapshot.takenAt.toLocaleString('zh-CN')}  |  阈值 ${snapshot.confidence}  |  ${snapshot.count} 个目标`, 18, canvas.height - 13);
  const capturedAt = snapshot.takenAt; canvas.toBlob(blob => {
    if (!blob) return error('截图保存失败，请重试'); const url = URL.createObjectURL(blob); const link = document.createElement('a');
    link.href = url; link.download = `ppe-${capturedAt.toISOString().replace(/[:.]/g,'-')}.png`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 30000);
  }, 'image/png');
}
el.startButton.addEventListener('click', start); el.stopButton.addEventListener('click', () => stop());
el.demoButton.addEventListener('click', demo); el.saveButton.addEventListener('click', save);
el.confidence.addEventListener('input', () => { el.confidenceValue.textContent = Number(el.confidence.value).toFixed(2); });
window.addEventListener('pagehide', () => stop());
async function connect() {
  try {
    const response = await fetch('/api/health'); if (!response.ok) throw new Error('本地模型连接失败'); const data = await response.json();
    ready = data.ready; modelLabel = data.model_label; el.modelLabel.textContent = modelLabel; el.readyDot.classList.add('ready');
    el.startButton.disabled = !ready; el.demoButton.dataset.available = String(data.demo_available); el.demoButton.disabled = !data.demo_available;
    el.frameStatus.textContent = '模型已就绪，点击右侧按钮开启摄像头';
  } catch (e) { el.modelLabel.textContent = '模型未连接'; el.frameStatus.textContent = '请先启动本地检测服务'; error(e.message); }
}
connect();
