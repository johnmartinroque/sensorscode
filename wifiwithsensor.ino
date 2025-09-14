#include <Wire.h>
#include "MAX30105.h"        // SparkFun MAX3010x library
#include "spo2_algorithm.h"  // Included in the same library
#include <WiFi.h>
#include <WebServer.h>

// ---------- WiFi Setup ----------
const char* ssid = "HUAWEI-2.4G-eAX8";   // Change to your WiFi SSID
const char* password = "pWfm5Aba";    // Change to your WiFi password

WebServer server(80);

// ---------- MAX30102 Setup ----------
MAX30105 particleSensor;

// Buffer size
#define BUFFER_SIZE 100
uint32_t irBuffer[BUFFER_SIZE];
uint32_t redBuffer[BUFFER_SIZE];

int32_t spo2;
int8_t validSPO2;
int32_t heartRate;
int8_t validHeartRate;

// ---------- Web Handler ----------
void handleRoot() {
  String html = "<!DOCTYPE html><html><head><meta http-equiv='refresh' content='2'>";
  html += "<title>ESP32-C3 Heart Monitor</title></head><body>";
  html += "<h1>ESP32-C3 MAX30102 Readings</h1>";
  if (validHeartRate && validSPO2) {
    html += "<p><b>BPM:</b> " + String(heartRate) + "</p>";
    html += "<p><b>SpO2:</b> " + String(spo2) + " %</p>";
  } else {
    html += "<p>Reading not valid yet...</p>";
  }
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  delay(2000);
  Serial.println("ESP32-C3 SuperMini Boot OK!");

  // ---------- Sensor Init ----------
  Wire.begin(6, 7);
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Please check wiring!");
    while (1);
  }

  particleSensor.setup(0x1F, 4, 2, 100, 411, 4096);
  Serial.println("MAX30102 initialization successful!");

  // Collect initial samples
  Serial.println("Place finger on sensor...");
  for (int i = 0; i < BUFFER_SIZE; i++) {
    while (!particleSensor.available()) particleSensor.check();
    redBuffer[i] = particleSensor.getRed();
    irBuffer[i] = particleSensor.getIR();
    particleSensor.nextSample();
    delay(10);
  }

  // ---------- WiFi Init ----------
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  int retry = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    if (++retry > 40) {
      Serial.println("\nFailed to connect to WiFi");
      return;
    }
  }
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());

  // Start server
  server.on("/", handleRoot);
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // ---------- Data Collection ----------
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

  // Run algorithm
  maxim_heart_rate_and_oxygen_saturation(
    irBuffer, BUFFER_SIZE,
    redBuffer,
    &spo2, &validSPO2,
    &heartRate, &validHeartRate);

  // Debug print
  if (validHeartRate && validSPO2) {
    Serial.print("BPM: ");
    Serial.print(heartRate);
    Serial.print("  |  SpO2: ");
    Serial.print(spo2);
    Serial.println("%");
  } else {
    Serial.println("Reading not valid yet...");
  }

  // ---------- Handle Web ----------
  server.handleClient();
}
