#include <Wire.h>
#include "MAX30105.h"     // Install SparkFun MAX3010x library
#include "heartRate.h"    // Comes with the same library

MAX30105 particleSensor;

void setup() {
  Serial.begin(115200);
  delay(2000);  
  Serial.println("ESP32-C3 SuperMini Boot OK!");

  // Initialize I2C with correct pins (SDA = GPIO6, SCL = GPIO7)
  Wire.begin(6, 7);

  // Run I2C Scanner
  Serial.println("Scanning I2C devices...");
  byte error, address;
  int nDevices = 0;
  for (address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    error = Wire.endTransmission();
    if (error == 0) {
      Serial.print("I2C device found at 0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      Serial.println(" !");
      nDevices++;
    }
  }
  if (nDevices == 0) Serial.println("No I2C devices found, check wiring!");

  // Initialize MAX30102
  Serial.println("Initializing MAX30102...");
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found. Please check wiring/power.");
    while (1);
  }

  particleSensor.setup(); // Configure with default settings
  particleSensor.setPulseAmplitudeRed(0x0A);   // Turn Red LED on
  particleSensor.setPulseAmplitudeGreen(0);    // Turn Green LED off
  Serial.println("MAX30102 initialization successful!");
}

void loop() {
  long irValue = particleSensor.getIR();
  long redValue = particleSensor.getRed();

  Serial.print("IR: ");
  Serial.print(irValue);
  Serial.print("\tRed: ");
  Serial.print(redValue);

  if (irValue < 50000) {
    Serial.println("\tNo finger detected");
  } else {
    if (checkForBeat(irValue)) {
      static unsigned long lastBeat = 0;
      static int beatsPerMinute;
      static int beatAvg;

      unsigned long delta = millis() - lastBeat;
      lastBeat = millis();

      beatsPerMinute = 60 / (delta / 1000.0);

      if (beatsPerMinute < 255 && beatsPerMinute > 20) {
        beatAvg = (beatAvg * 3 + beatsPerMinute) / 4;
        Serial.print("\tBPM: ");
        Serial.println(beatAvg);
      }
    } else {
      Serial.println();
    }
  }

  delay(100);
}
