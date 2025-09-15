// ==========================
// ESP32-C3 GSR + MAX30102 Web Dashboard
// ==========================

// ---------- Includes ----------
#include <WiFi.h>
#include <WebServer.h>
#include <Wire.h>
#include "MAX30105.h"        // SparkFun MAX3010x library
#include "spo2_algorithm.h"  // Comes with SparkFun library

// ---------- WiFi ----------
const char* ssid = "HUAWEI-2.4G-eAX8";
const char* password = "pWfm5Aba";

WebServer server(80);

// ---------- GSR Setup ----------
#define GSR_PIN 4
const float VCC = 3.3f;
const int ADC_MAX = 4095;

const int MA_SIZE = 8;
int maBuf[MA_SIZE];
int maIdx = 0;
long maSum = 0;

float rawToVoltage(int raw) {
  return (raw / (float)ADC_MAX) * VCC;
}

// ---------- MAX30102 Setup ----------
MAX30105 particleSensor;

#define BUFFER_SIZE 100
uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];

int32_t spo2;
int8_t validSPO2;
int32_t heartRate;
int8_t validHeartRate;

// ---------- Webpage ----------
String webpage = R"rawliteral(
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ESP32-C3 Health Monitor</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:18px;background:#111;color:#eee}
    .card{background:#1b1b1b;padding:14px;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.5);margin-bottom:16px}
    h1{margin:0 0 8px 0;font-size:20px}
    #val{font-size:18px;font-weight:600}
    #vol{color:#bbb}
    canvas{width:100%;height:80px;background:#0f0f0f;border-radius:6px;margin-top:10px}
  </style>
</head>
<body>
  <div class="card">
    <h1>GSR Live</h1>
    <p id="status">Connecting...</p>
    <div>
      <div id="val">Raw: --</div>
      <div id="vol">Voltage: -- V</div>
    </div>
    <canvas id="chart"></canvas>
  </div>

  <div class="card">
    <h1>Heart Monitor</h1>
    <div id="hr">BPM: --</div>
    <div id="spo2">SpO₂: -- %</div>
  </div>

<script>
const pollInterval = 500; // ms
let buffer = [];
const maxPoints = 70;

function updateUI(raw, avg, v, bpm, spo2) {
  document.getElementById('status').innerText = 'Connected';
  document.getElementById('val').innerText = `Raw: ${raw}  |  Avg: ${avg}`;
  document.getElementById('vol').innerText = `Voltage: ${v.toFixed(3)} V`;
  document.getElementById('hr').innerText = `BPM: ${bpm > 0 ? bpm : '--'}`;
  document.getElementById('spo2').innerText = `SpO₂: ${spo2 > 0 ? spo2 : '--'} %`;

  buffer.push(avg);
  if (buffer.length > maxPoints) buffer.shift();
  drawSparkline();
}

function drawSparkline(){
  const canvas = document.getElementById('chart');
  const ctx = canvas.getContext('2d');
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
    const resp = await fetch('/data');
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const j = await resp.json();
    updateUI(j.raw, j.avg, j.voltage, j.heartRate, j.spo2);
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

// ---------- Handlers ----------
void handleRoot() {
  server.send(200, "text/html", webpage);
}

void handleData() {
  int raw = analogRead(GSR_PIN);
  maSum -= maBuf[maIdx];
  maBuf[maIdx] = raw;
  maSum += maBuf[maIdx];
  maIdx = (maIdx + 1) % MA_SIZE;
  int avg = maSum / MA_SIZE;
  float v = rawToVoltage(avg);

  String payload = "{";
  payload += "\"raw\":" + String(raw) + ",";
  payload += "\"avg\":" + String(avg) + ",";
  payload += "\"voltage\":" + String(v, 4) + ",";
  payload += "\"heartRate\":" + String(validHeartRate ? heartRate : -1) + ",";
  payload += "\"spo2\":" + String(validSPO2 ? spo2 : -1);
  payload += "}";
  server.send(200, "application/json", payload);
}

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("ESP32-C3 Health Monitor Starting...");

  for (int i=0;i<MA_SIZE;i++){ maBuf[i] = 0; maSum += maBuf[i]; }

  // MAX30102 Init
  Wire.begin(6, 7);
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }
  particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);

  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
    delay(10);
  }

  // WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  int attempt = 0;
  while (WiFi.status() != WL_CONNECTED && attempt < 50) {
    delay(400);
    Serial.print(".");
    attempt++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi not connected!");
  }

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.begin();
  Serial.println("HTTP server started");
}

// ---------- Loop ----------
void loop() {
  // Heart Rate sampling
  for (int i = 25; i < BUFFER_SIZE; i++) {
    redBuffer[i - 25] = redBuffer[i];
    irBuffer[i - 25] = irBuffer[i];
  }
  for (int i = BUFFER_SIZE - 25; i < BUFFER_SIZE; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
    delay(10);
  }
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_SIZE,
    redBuffer,
    &spo2, &validSPO2,
    &heartRate, &validHeartRate);

  server.handleClient();
  delay(10);
}
