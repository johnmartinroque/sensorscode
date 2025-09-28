#include <WiFi.h>
#include <WebServer.h>  // Built-in lightweight HTTP server

const char* ssid = "wifiname";
const char* password = "pass";

// Create a web server on port 80
WebServer server(80);

void handleRoot() {
  server.send(200, "text/plain", "Hello, world");
}

void setup() {
  Serial.begin(115200);
  delay(1000);

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

  // Define what happens when someone accesses "/"
  server.on("/", handleRoot);

  // Start the server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  server.handleClient();  // Handle incoming client requests
}
