// ESP32-C3 GSR Live Webpage (AJAX polling, no page refresh)
// SIG -> GPIO4, VCC->3.3V, GND->GND
// Update: ssid/password below

#include <WiFi.h>
#include <WebServer.h>

const char* ssid = "HUAWEI-2.4G-eAX8";
const char* password = "pWfm5Aba";

#define GSR_PIN 4
const float VCC = 3.3f;
const int ADC_MAX = 4095;

// moving average
const int MA_SIZE = 8;
int maBuf[MA_SIZE];
int maIdx = 0;
long maSum = 0;

WebServer server(80);

String webpage = R"rawliteral(
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>GSR Live</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:18px;background:#111;color:#eee}
    .card{background:#1b1b1b;padding:14px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.5)}
    h1{margin:0 0 8px 0;font-size:20px}
    #val{font-size:20px;font-weight:600}
    #vol{color:#bbb}
    canvas{width:100%;height:80px;background:#0f0f0f;border-radius:6px;margin-top:10px}
  </style>
</head>
<body>
  <div class="card">
    <h1>GSR Live (ESP32-C3)</h1>
    <p id="status">Connecting...</p>
    <div>
      <div id="val">Raw: --</div>
      <div id="vol">Voltage: -- V</div>
    </div>
    <canvas id="chart"></canvas>
  </div>

<script>
const pollInterval = 300; // ms
let buffer = [];
const maxPoints = 70;

function updateUI(raw, avg, v) {
  document.getElementById('status').innerText = 'Connected';
  document.getElementById('val').innerText = `Raw: ${raw}  |  Avg: ${avg}`;
  document.getElementById('vol').innerText = `Voltage: ${v.toFixed(3)} V`;

  // sparkline buffer
  buffer.push(avg);
  if (buffer.length > maxPoints) buffer.shift();
  drawSparkline();
}

function drawSparkline(){
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
  // dynamic resize
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,canvas.width,canvas.height);

  if (buffer.length === 0) return;
  const w = canvas.width, h = canvas.height;
  const minV = Math.min(...buffer);
  const maxV = Math.max(...buffer);
  const range = (maxV - minV) || 1;

  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.beginPath();
  for (let i=0;i<buffer.length;i++){
    const x = (i/(buffer.length-1)) * (w - 6*devicePixelRatio) + 3*devicePixelRatio;
    const y = h - ((buffer[i]-minV)/range) * (h - 6*devicePixelRatio) - 3*devicePixelRatio;
    if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.strokeStyle = '#00ffcc';
  ctx.stroke();
}

async function poll(){
  try {
    const resp = await fetch('/gsr');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const j = await resp.json();
    updateUI(j.raw, j.avg, j.voltage);
  } catch (e) {
    document.getElementById('status').innerText = 'Error: ' + e.message;
  } finally {
    setTimeout(poll, pollInterval);
  }
}

poll();
</script>
</body>
</html>
)rawliteral";

void handleRoot() {
  server.send(200, "text/html", webpage);
}

float rawToVoltage(int raw) {
  return (raw / (float)ADC_MAX) * VCC;
}

void handleGSR() {
  // return latest sample as JSON
  int raw = analogRead(GSR_PIN);
  // update moving average
  maSum -= maBuf[maIdx];
  maBuf[maIdx] = raw;
  maSum += maBuf[maIdx];
  maIdx = (maIdx + 1) % MA_SIZE;
  int avg = maSum / MA_SIZE;
  float v = rawToVoltage(avg);

  String payload = "{";
  payload += "\"raw\":" + String(raw) + ",";
  payload += "\"avg\":" + String(avg) + ",";
  payload += "\"voltage\":" + String(v, 4);
  payload += "}";
  server.send(200, "application/json", payload);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  Serial.println("GSR Live Webpage");

  // init MA buffer
  for (int i=0;i<MA_SIZE;i++){ maBuf[i] = 0; maSum += maBuf[i]; }

  // ADC attenuation: allow full 0-3.3V (uncomment if needed)
  // analogSetPinAttenuation(GSR_PIN, ADC_11db);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int attempt = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
    attempt++;
    if (attempt >= 50) break; // don't block forever
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println();
    Serial.println("WiFi not connected - AP mode not implemented in this sketch.");
  }

  server.on("/", handleRoot);
  server.on("/gsr", handleGSR);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();
  // small delay to yield CPU
  delay(10);
}
