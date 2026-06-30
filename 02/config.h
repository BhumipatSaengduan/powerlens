#pragma once

// ═══════════════════════════════════════════════
//  PowerLens Sensor Firmware — config.h
//  Device: SM-PE-02 (TESTER)
//  Generated for: Prism Energy Co., Ltd
// ═══════════════════════════════════════════════

// ── Device Identity ──────────────────────────────
#define DEVICE_ID        "SM-PE-02"
#define SERIAL_NO        "A01-A-2026-F01-T00002"
#define SITE_ID          "SITE-001"
#define FW_VERSION       "1.1.0-local-recorder"
#define DEVICE_TYPE      "TESTER"

// One-shot reset token for rate schedule experiment_start.
// Bump this value when you want a new "Day 1" after reflashing.
#define EXPERIMENT_START_RESET_TOKEN 20260526UL

// ── WiFi ─────────────────────────────────────────
#define WIFI_SSID        "TP-Link_HA"
#define WIFI_PASSWORD    "Energyhub123+"

// ── AWS IoT ──────────────────────────────────────
#define AWS_ENDPOINT     "ad6vdej5wue1t-ats.iot.ap-southeast-1.amazonaws.com"
#define AWS_PORT         8883

#define MQTT_TOPIC_TELEMETRY  "powerlens/" SITE_ID "/" DEVICE_ID "/telemetry"
#define MQTT_TOPIC_STATUS     "powerlens/" SITE_ID "/" DEVICE_ID "/status"
#define MQTT_TOPIC_LWT        "powerlens/" SITE_ID "/" DEVICE_ID "/status"

// ── AWS Certificates ─────────────────────────────
// วาง cert จากไฟล์เหล่านี้:
//   ~/Desktop/pem/SM-PE-02-cert.pem      → AWS_CERT_CRT
//   ~/Desktop/pem/SM-PE-02-private.key   → AWS_CERT_PRIVATE
//   ~/Desktop/pem/AmazonRootCA1.pem      → AWS_CERT_CA
//
// รันใน Terminal:
//   cat ~/Desktop/pem/AmazonRootCA1.pem
//   cat ~/Desktop/pem/SM-PE-02-cert.pem
//   cat ~/Desktop/pem/SM-PE-02-private.key

const char AWS_CERT_CA[] = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
ADA5MQswCQYDVQQGEwJVUzEPMA0GA1UEChMGQW1hem9uMRkwFwYDVQQDExBBbWF6
b24gUm9vdCBDQSAxMB4XDTE1MDUyNjAwMDAwMFoXDTM4MDExNzAwMDAwMFowOTEL
MAkGA1UEBhMCVVMxDzANBgNVBAoTBkFtYXpvbjEZMBcGA1UEAxMQQW1hem9uIFJv
b3QgQ0EgMTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALJ4gHHKeNXj
ca9HgFB0fW7Y14h29Jlo91ghYPl0hAEvrAIthtOgQ3pOsqTQNroBvo3bSMgHFzZM
9O6II8c+6zf1tRn4SWiw3te5djgdYZ6k/oI2peVKVuRF4fn9tBb6dNqcmzU5L/qw
IFAGbHrQgLKm+a/sRxmPUDgH3KKHOVj4utWp+UhnMJbulHheb4mjUcAwhmahRWa6
VOujw5H5SNz/0egwLX0tdHA114gk957EWW67c4cX8jJGKLhD+rcdqsq08p8kDi1L
93FcXmn/6pUCyziKrlA4b9v7LWIbxcceVOF34GfID5yHI9Y/QCB/IIDEgEw+OyQm
jgSubJrIqg0CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
AYYwHQYDVR0OBBYEFIQYzIU07LwMlJQuCFmcx7IQTgoIMA0GCSqGSIb3DQEBCwUA
A4IBAQCY8jdaQZChGsV2USggNiMOruYou6r4lK5IpDB/G/wkjUu0yKGX9rbxenDI
U5PMCCjjmCXPI6T53iHTfIUJrU6adTrCC2qJeHZERxhlbI1Bjjt/msv0tadQ1wUs
N+gDS63pYaACbvXy8MWy7Vu33PqUXHeeE6V/Uq2V8viTO96LXFvKWlJbYK8U90vv
o/ufQJVtMVT8QtPHRh8jrdkPSHCa2XV4cdFyQzR1bldZwgJcJmApzyMZFo6IQ6XU
5MsI+yMRQ+hDKXJioaldXgjUkK642M4UwtBV8ob2xJNDd2ZhwLnoQdeXeGADbkpy
rqXRfboQnoZsG4q5WTP468SQvvG5
-----END CERTIFICATE-----
)EOF";

const char AWS_CERT_CRT[] = R"EOF(
-----BEGIN CERTIFICATE-----
MIIDWTCCAkGgAwIBAgIUQFQ5Wicy4r/fMTKETrcH0loZb60wDQYJKoZIhvcNAQEL
BQAwTTFLMEkGA1UECwxCQW1hem9uIFdlYiBTZXJ2aWNlcyBPPUFtYXpvbi5jb20g
SW5jLiBMPVNlYXR0bGUgU1Q9V2FzaGluZ3RvbiBDPVVTMB4XDTI2MDUxMTAzNDQw
NFoXDTQ5MTIzMTIzNTk1OVowHjEcMBoGA1UEAwwTQVdTIElvVCBDZXJ0aWZpY2F0
ZTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKoW6uJgL0zdH/1MaKs+
gvuoj2R8+dgJoEZ3qw9ZsnQnl7fntI6ULsZlrli/TKFj6IGpkAIT7PYIEh9zpSW4
rvRBrfE/wUjeL2Cc54ok23614Nx1IC/V+PCTLnpvVIwAFusvG+pUDWrVbG9x1GpR
O9cPriKMQhCRFD6g8IgsLqFXxu3gJUbKKZBrDwkc9mLmmkZJ/TSMwdg70to8Ef2b
EHgfKn2faz+amOmF83HxS/CGd7NyBO1VXGtirkcuHTCqzVTbrPeZxp/iR/knMjie
Fi/HMDBaL+Djk16oxKIIyEFF/bpRuVxxyJcRJbDoB24QtTGrR1+JpJeyRn5YKIgd
29UCAwEAAaNgMF4wHwYDVR0jBBgwFoAUp+74jc91sTD6epvlZ6d/oUctiO4wHQYD
VR0OBBYEFCTixmaMFzlWv50liaUVD2807XsVMAwGA1UdEwEB/wQCMAAwDgYDVR0P
AQH/BAQDAgeAMA0GCSqGSIb3DQEBCwUAA4IBAQCRiqU9e3XjJ3NdmQZeBAnOpzkW
PWii+KSvSUUa7xq0mHxptmDfcbTO5yvNNYI4hcOyVSt+1hvat3EIErmM+RuES/9W
+0J4TVdIi1nJFX5wxwa9A+csckV1bSOOQ3vxIdLNjuxPG4tcpfkmRgWgf3mnIYC0
DPnkbmbgvroGGLT8EvxkGxP86XkbHPfebWIuDBdB3PBOt0/0DJAsxKZ+vMRC8maT
O19iS5XEjgS234348xaFX9wtvSssQa3NuLzKVqovm7iy+fFdYsr+aTnMxuBtcOLM
15S1otkj5Zdgbap2zMeAlgdh93EB8dGWSRTAeVYpvCwYMJcOtFs8z7jKpUJO
-----END CERTIFICATE-----
)EOF";

const char AWS_CERT_PRIVATE[] = R"EOF(
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAqhbq4mAvTN0f/Uxoqz6C+6iPZHz52AmgRnerD1mydCeXt+e0
jpQuxmWuWL9MoWPogamQAhPs9ggSH3OlJbiu9EGt8T/BSN4vYJzniiTbfrXg3HUg
L9X48JMuem9UjAAW6y8b6lQNatVsb3HUalE71w+uIoxCEJEUPqDwiCwuoVfG7eAl
RsopkGsPCRz2YuaaRkn9NIzB2DvS2jwR/ZsQeB8qfZ9rP5qY6YXzcfFL8IZ3s3IE
7VVca2KuRy4dMKrNVNus95nGn+JH+ScyOJ4WL8cwMFov4OOTXqjEogjIQUX9ulG5
XHHIlxElsOgHbhC1MatHX4mkl7JGflgoiB3b1QIDAQABAoIBAEsV0NzcPyU8XHnS
OEaYUvRLZfmjXhvrq/BPtZkSLMAwFj7eL4vdiISWsI+G64o3c5WByAvSxgGacH2n
7JipXbqAIAxm66mCRAHvYhtOyAK4waownmPfnoR9RMBR0032YCe0ZStdrYqi6rqL
0oyYjcUTq/ieWC++C+TV4TkL4A3i3LV6edce8b5w3wTcEOmrEWIuOo/arwds1P/V
MdV5oteJCGxl+ziPX1WfiUzlywprkLxc7Ei66/e5tT3VS1M50jtJovFsTZbRzvjZ
fA/twZviK0kBiHX1eyuSYpa15szvprZHJtrvx16L7iNnuVrPRqpscPP3XtpP3R6b
fzx+wEECgYEA0oIuuP7sMKrPxXXQN2J2YM8JAefNCI6SOZMBvlgKD8w8LdCG5i/o
szZaqpwU9qNvrMWxT7Y0Yl0VyhMgpd1eMRox75yWZI+Ko9381Ez9zBrYL4kjR7Cx
cbgGePTuoFJ9KcIhvJXKNJXv173D188m7REBC/13bRaLqj7EHm+7oDECgYEAztip
lIutFa/dBRbcUTpTi6rhVsa77cJb8rtq6KJb6rUkuVNJj/L+V3L9tLJMEwpme27k
kNCrKOmboH0es61nVoTDQdGypeD0USBooG6WbV+S54fEnSN7GJ/1Ca6jMv63he8X
nfiL5ZqLj86mmXIdpfTxYWt3Lm/8qfouib7skOUCgYEAiC9MRsY0yu9WZypmv83l
Q4/tBdyOWoDRvImMUTXnnHzGWeVTwEsyQe5iDYnYTg9BygZDRYxcq14JIKfrMSLb
Muz9bURiT0BFsumEDVyZvJeJUIdp2ZFH2ofxOANM9U8oRgGfjb9iB08Q0QOlVVJg
nnGnubgKsPoq9MKSYhZqzaECgYBOfa8UHFCo6xw+wycFd9GeLVDnIfDMTzWfDXmL
H5krnmN6I93FTxsuygb2G7Z8fzTWYAVB4r0ggE07AF+3JPUSwrxpbI6THaL4agjp
4C0bAep4C3AThRRACursKqXpQvkXTNw0aM2FajjNcEiN79zKTgGOyz3llD9XrQUd
5iJU9QKBgA7HzIDONTpAjDr4qeTzFyaeiz6VTej2vHC+ON45en4VGwvwtnnt/uLL
mtgyEoP/9MrtMubZ3fVEGGiBfAVzQdQonMEP1iidwgxJnnna6bl0Q053m77U/Bxv
kuTz1Gr9F99tEJpbhX+jU7C3uq54bp9I/hFFa0y/w+GRgt3t20MT
-----END RSA PRIVATE KEY-----
)EOF";

// ── UART (รับจาก Pi Pico 2W) ─────────────────────
//  Verified working wiring matches SM-PE-01 field test:
//      ESP32 GPIO 18 (RX) ←─────── Pi GPIO16 (TX)
//      ESP32 GPIO 17 (TX) ────────→ Pi GPIO17 (RX)
//      common GND
#define UART_BAUD        2000000   // 2 Mbps
#define UART_RX_PIN      18        // ต่อกับ Pi GPIO16 (TX)
#define UART_TX_PIN      17        // ต่อกับ Pi GPIO17 (RX)
#define UART_BUF_SIZE    8192

// ── LED Status Mapping (forwarded to Pi only; ESP32 does not drive local GPIO) ──
// Pi LED1 = power, Pi LED2 = WiFi link, Pi LED3 = successful SD raw write.
#define LED_POWER        4    // semantic mapping for Pi LED1: power
#define LED_STATUS       5    // semantic mapping for Pi LED2: WiFi connected
#define LED_DATA         19   // semantic mapping for Pi LED3: SD raw write success pulse
#define DATA_LED_PULSE_MS 80UL // short pulse so 4 raw chunks/sec still looks like blinking

// ── Packet Protocol ──────────────────────────────
//  หมายเหตุ: ค่าเหล่านี้สงวนไว้สำหรับ future use — ยังไม่ถูก reference ในโค้ดปัจจุบัน
//  โค้ดที่ใช้งานจริง:
//      RAW chunk header magic = 0x504C  ('PL')        — ดู RawChunkHeader
//      Metrics packet bracket = 0xAA / 0x55           — ดู METRICS_START_BYTE / END_BYTE
//      Rate-set packet bracket= 0xFC / 0xFD           — ดู RATE_SET_PACKET_START / END
//      RAW chunk samples      = 250 (= RAW_CHUNK_SAMPLES) ส่งทุก 250ms
//      Sample rate            = SAMPLE_RATE_HZ ใน main file
#define PKT_START        0xAA       // (reserved for future custom protocol)
#define PKT_END          0x55       // (reserved for future custom protocol)
#define N_CHANNELS       6          // V1, I1, V2, I2, V3, I3
#define SAMPLES_PER_PKT  100        // (reserved — actual is RAW_CHUNK_SAMPLES = 250)

// ── Timing ───────────────────────────────────────
#define STATUS_INTERVAL_MS    30000   // heartbeat ทุก 30 วินาที
#define OFFLINE_TIMEOUT_MS    600000  // alert ถ้าไม่ส่ง 10 นาที
#define SAMPLE_INTERVAL_US    200     // (reserved — actual is SAMPLE_PERIOD_US in main file)

// ── Temperature Sensor (DS18B20, 1-Wire on ESP32 GPIO32) ───────────────────
#define TEMP_SENSOR_PIN     32
#define TEMP_READ_INTERVAL  5000

// ── SD Card Local Logging (ESP32 SPI, non-blocking writer task) ─────────────
//  Production raw-only mode: SD เก็บ raw binary; AWS ใช้ status สำหรับ monitoring
//  ไม่ขอ/ไม่ส่ง Pi metrics เพื่อไม่ให้ feature compute แทรก ADC loop
//  หลีกเลี่ยง GPIO17/18 เพราะใช้ UART กับ Pi อยู่แล้ว
#define ENABLE_SD_CARD_LOG      1
#define ENABLE_RAW_SD_LOG       1
#define ENABLE_FEATURE_SD_LOG   0
#define ENABLE_COMBINED_SD_LOG  0
#define ENABLE_RAW_UPLOAD       0
#define ENABLE_PI_METRICS_PACKET 0
// Production local-recorder mode: no TLS, MQTT, or AWS publish task is started.
// WiFi remains enabled only for NTP and the front-panel WiFi LED.
#define ENABLE_AWS_IOT          0

#define SD_CARD_CS_PIN          26 
#define SD_CARD_SCK_PIN         14
#define SD_CARD_MISO_PIN        25 
#define SD_CARD_MOSI_PIN        27
#define SD_CARD_SPI_FREQ        4000000UL
#define SD_ROOT_DIR             "/powerlens"

// 8 buffers ≈ 2 วินาทีของ raw chunk ที่ 1 kHz และคืน heap ให้ TLS
#define RAW_BUFFER_COUNT        8
#define SD_RAW_FLUSH_INTERVAL_CHUNKS     8
#define SD_FEATURE_FLUSH_INTERVAL_ROWS   10
#define SD_FEATURE_QUEUE_DEPTH           16
