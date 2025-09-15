#include <WiFi.h>
#include <WebServer.h>  // Built-in lightweight HTTP server

const char* ssid = "HUAWEI-2.4G-eAX8";
const char* password = "pWfm5Aba";

// Create a web server on port 80
WebServer server(80);

void handleRoot() {
  server.sendHeader("Access-Control-Allow-Origin", "*");   // allow all origins
  server.sendHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  server.sendHeader("Access-Control-Allow-Headers", "*");
  server.send(200, "text/plain", "Hello, world");
}


void setup() {
  Serial.begin(115200);
  delay(2000);  // Allow time for USB CDC

  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 40) {
    delay(500);
    Serial.print(".");
    retry++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // Define what happens when someone accesses "/"
    server.on("/", handleRoot);

    // Start the server
    server.begin();
    Serial.println("HTTP server started");
  } else {
    Serial.println("\nFailed to connect to WiFi");
  }
}

void loop() {
  server.handleClient();  // Handle incoming client requests
}
