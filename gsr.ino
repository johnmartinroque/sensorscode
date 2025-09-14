#include <Arduino.h>

// Define GSR pin (use an ADC pin on ESP32-C3, e.g., GPIO4 or GPIO5)
// GPIO6 and GPIO7 are for I2C – better avoid those for analog
#define GSR_PIN 4  

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("GSR Sensor Test Start");
}

void loop() {
  int gsrValue = analogRead(GSR_PIN);   // Read analog value from GSR
  float voltage = (gsrValue / 4095.0) * 3.3; // Convert to voltage (0–3.3V)
  
  Serial.print("GSR Value: ");
  Serial.print(gsrValue);
  Serial.print("  | Voltage: ");
  Serial.println(voltage, 3);

  delay(500); // Small delay for readability
}
