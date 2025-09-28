// GSR for ESP32-C3 SuperMini
// SIG -> ADC pin (default GPIO4). VCC->3.3V, GND->GND, NC ignored.
//
// Upload and open Serial Monitor at 115200. Press 'c' to calibrate baseline (no-contact).
// If you know your module's pull resistor (ohms), set RPULL_OHMS to that value to get Rskin/G.

#include <Arduino.h>

// --- Configuration ---
#define GSR_PIN     4           // SIG -> GPIO4 (change if you used a different ADC pin)
const float VCC = 3.3f;        // sensor power
const int ADC_MAX = 4095;      // default 12-bit ADC on ESP32 cores

// Optional: set the fixed resistor value used by your GSR module in ohms.
// If unknown, set to 0.0 and resistance/conductance will be skipped.
const float RPULL_OHMS = 0.0f; // e.g. 1000000.0 for 1M, 10000.0 for 10k. Set to 0.0 if unknown.

// Moving average
const int MA_SIZE = 16;
int maBuffer[MA_SIZE];
int maIndex = 0;
long maSum = 0;

// Calibration baseline (raw and averaged)
int baselineRaw = 0;
float baselineVoltage = 0.0f;

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("GSR - ESP32-C3 SuperMini");
  Serial.println("Press 'c' to calibrate baseline (no contact).");

  // initialize buffer
  for (int i = 0; i < MA_SIZE; ++i) {
    maBuffer[i] = 0;
  }

  // Optional: set attenuation for the ADC pin to allow full 0-3.3V reading
  // (options: ADC_0db, ADC_2_5db, ADC_6db, ADC_11db)
  // analogSetPinAttenuation(GSR_PIN, ADC_11db); // uncomment if needed
}

float rawToVoltage(int raw) {
  return (raw / (float)ADC_MAX) * VCC;
}

float calcResistanceFromVout(float vout) {
  // assumes simple divider: Vout = Vcc * (Rskin / (Rskin + Rpull)) => Rskin = Rpull * (Vout / (Vcc - Vout))
  if (RPULL_OHMS <= 0.0f) return -1.0f;
  if (vout <= 0.0f || vout >= VCC) return -1.0f;
  return RPULL_OHMS * (vout / (VCC - vout));
}

void updateMA(int value) {
  maSum -= maBuffer[maIndex];
  maBuffer[maIndex] = value;
  maSum += maBuffer[maIndex];
  maIndex = (maIndex + 1) % MA_SIZE;
}

int getMA() {
  return (int)(maSum / MA_SIZE);
}

void loop() {
  // read raw
  int raw = analogRead(GSR_PIN);
  updateMA(raw);
  int avg = getMA();
  float vout = rawToVoltage(avg);

  // optional: resistance / conductance
  float rskin = calcResistanceFromVout(vout); // ohms (-1 if unknown or invalid)
  float conductance = (rskin > 0.0f) ? (1.0f / rskin) : -1.0f;

  // If baseline not set, we still show raw/voltage
  Serial.print("Raw: ");
  Serial.print(raw);
  Serial.print(" | Avg: ");
  Serial.print(avg);
  Serial.print(" | V: ");
  Serial.print(vout, 3);

  if (RPULL_OHMS > 0.0f) {
    if (rskin > 0.0f) {
      Serial.print(" | Rskin(ohm): ");
      Serial.print(rskin, 0);
      Serial.print(" | G(uS): ");
      Serial.print(conductance * 1e6, 2); // microsiemens
    } else {
      Serial.print(" | Rskin: N/A");
    }
  }

  // If baseline calibrated show relative change %
  if (baselineRaw > 0) {
    float pct = 100.0f * (avg - baselineRaw) / (float)baselineRaw;
    Serial.print(" | Δ%: ");
    Serial.print(pct, 2);
  }

  Serial.println();

  // calibration command (press 'c' in serial monitor)
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C') {
      // take a short average for baseline
      long sum = 0;
      const int N = 64;
      for (int i = 0; i < N; ++i) {
        sum += analogRead(GSR_PIN);
        delay(5);
      }
      baselineRaw = (int)(sum / N);
      baselineVoltage = rawToVoltage(baselineRaw);
      Serial.print("Baseline calibrated: raw=");
      Serial.print(baselineRaw);
      Serial.print(" | V=");
      Serial.println(baselineVoltage, 3);
    }
  }

  delay(120); // ~8-9 samples/sec - change as needed
}
