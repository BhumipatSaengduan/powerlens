#pragma once

// ═══════════════════════════════════════════════
//  PowerLens Sensor Firmware — config.h
//  Device: SM-PE-03 (TESTER)
//  Generated for: Prism Energy Co., Ltd
// ═══════════════════════════════════════════════

// ── Device Identity ──────────────────────────────
#define DEVICE_ID        "SM-PE-03"
#define SERIAL_NO        "A01-A-2026-F01-T00003"
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
//   ~/Desktop/pem/SM-PE-03-cert.pem      → AWS_CERT_CRT
//   ~/Desktop/pem/SM-PE-03-private.key   → AWS_CERT_PRIVATE
//   ~/Desktop/pem/AmazonRootCA1.pem      → AWS_CERT_CA
//
// รันใน Terminal:
//   cat ~/Desktop/pem/AmazonRootCA1.pem
//   cat ~/Desktop/pem/SM-PE-03-cert.pem
//   cat ~/Desktop/pem/SM-PE-03-private.key

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
MIIDWTCCAkGgAwIBAgIUefTPoHjCFjJQifcmACQmto/YHMswDQYJKoZIhvcNAQEL
BQAwTTFLMEkGA1UECwxCQW1hem9uIFdlYiBTZXJ2aWNlcyBPPUFtYXpvbi5jb20g
SW5jLiBMPVNlYXR0bGUgU1Q9V2FzaGluZ3RvbiBDPVVTMB4XDTI2MDQxNzA0MTUz
OVoXDTQ5MTIzMTIzNTk1OVowHjEcMBoGA1UEAwwTQVdTIElvVCBDZXJ0aWZpY2F0
ZTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAN1b25LKF15Dhvn9mlcs
KDkmok8mrZ4Nu5naSx1vfRoYR6GMcVgn1tV77IYasv/PasT3TDo6DsMbiN9su7+M
7EQME0Yjf+RK8BzKsEvvXA1r/RsFMytQkl/ksHGH4EUN9xLNudVdoQTaSSSvs3Jl
e2B1B+DDJlj4w8wkRiBjphxaeFwjpMybOn+JiDjf8Q76n60TnJfT8PIvYBSpOQzn
cljjGXWKz9YVCmAsnEW+9S3c/Lct3B5/vqcDgbjgvLTcz0ywZUGlL3kP1gTfPTAt
DQ/UmQkkV5a4nHA8LBZWHOwSeHSlF10Y38VV0hzcD4q6i41u9LUpdmTbacFzgmUb
uMUCAwEAAaNgMF4wHwYDVR0jBBgwFoAUUEIthr2fXmfUidVokerbiII3bYkwHQYD
VR0OBBYEFCWoTFmEo+Z2L/bcUhAHoedEvyK6MAwGA1UdEwEB/wQCMAAwDgYDVR0P
AQH/BAQDAgeAMA0GCSqGSIb3DQEBCwUAA4IBAQAPb4nccUjFzK2KqwXsXblD/auX
Lq6V6da4SktNCV8UvNhS6h22Edc6ybTiPHK1KRILBVX2T3dW9ffaB3hxu/ht7hG2
K7dlU6BQpbkzA5lZu42TQt1MUwepWg01hoylLP8tA89inEpF6NML21WzGXho26VS
ZUUhnuWp+cITmWyQminrRBTLf5F4QznTBVBWwxuU9XHsVm3ZkRpPFPRvz6sKI0Uf
JgbYRMno+1v2dPU6tejas9f1LYrUekuqe+kzD/lWNbr40HwYkzBj+odi1aa36nSE
/5v9DwFQ/QtJIBrpY2JIvZKL5NnILEoQtl2TBKYrAHZ5R1uJCyPPLmYXYZ4N
-----END CERTIFICATE-----
)EOF";

const char AWS_CERT_PRIVATE[] = R"EOF(
-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA3VvbksoXXkOG+f2aVywoOSaiTyatng27mdpLHW99GhhHoYxx
WCfW1Xvshhqy/89qxPdMOjoOwxuI32y7v4zsRAwTRiN/5ErwHMqwS+9cDWv9GwUz
K1CSX+SwcYfgRQ33Es251V2hBNpJJK+zcmV7YHUH4MMmWPjDzCRGIGOmHFp4XCOk
zJs6f4mION/xDvqfrROcl9Pw8i9gFKk5DOdyWOMZdYrP1hUKYCycRb71Ldz8ty3c
Hn++pwOBuOC8tNzPTLBlQaUveQ/WBN89MC0ND9SZCSRXlriccDwsFlYc7BJ4dKUX
XRjfxVXSHNwPirqLjW70tSl2ZNtpwXOCZRu4xQIDAQABAoIBACyZGUAV32dqi4NK
iRIPH3uEQSdZT5mMgsOYq5GeqMHdKnFt7lgojqwsb5cFQhMwIv7UJFOG5vqATa9W
JO2O4vtCw49aD6ZbQs1KOQLTkuRRYYlUHt8XOKaBWNtG2PSQv7rWIB7Q4mQr5pix
naHquFTSv2eVaeB7Hle+5zIXYZxMbg02WQKpg3C5FD2hHrhNeWkj7LNsvCQVss1V
rtZ4NmgrpTcXaVM20R8man0jttNae3zxaP4bYYFs9AmfQNllBMiBAlxYAfVN9g7d
PoTSHFh8HX1sabz0LGwT4tVG76nWRHy2TwHQj/bapwTVOi1DCCyYwXcvCib/ILNF
36gc6fECgYEA8Q+UVLj2O0psZL4qccOd9AVhIlgVQ8899zRIA9HVBjbU51QOZiMg
hA/r0Ga/VHgX0IeB4YTD421MUXKD3JFM2oh6nu1Yl+3FCNvU+9QTCx3e/SZfGGma
7NEwK9yu3IKQX2wJk7OnKe1m8Jr9BJfkwFrTH9brF5hcYvTww/ui6I8CgYEA6xO0
zhRfAQZ+rj7P74/98i1LOEO/7PI0W93Po5Wr4Z4Fj4c47zFwreeRB/uSlwi5y31c
mvzqeQdWJG/d879ARInHSOasA6BjUf3HiJcbTanTriiH13Zdo9QWLmx+SSkaNdNG
XgvxoxZP3MleoquKikssvJhv0p9WAd3VIAMDq2sCgYEAhNUR5nGZdWh4PDcxykiB
tGJ2eOdSqG/9dEfB2yD4Ipl6ThJacNuwYjUnu0my6ofWj5jr7+opyxuCL2tLz/Hd
CJql/wdIh0eFCHGidjYRXFHUe2h2hExFC3Pl+HV9gZMMnRg6WsJnPcpMrA6rl6lf
asNhjSzvlKvnVLwmI8h4p28CgYAZiQ9331CenOT/6oTN4hdUykTEfN+JTpoPbJ3U
iDqejHrQJ4Ewwm8aBPCFLHe0/laoWxrHHzgdI4Xg+WHjy1+g0lKagawnzLFdQ7L8
DGYD3rHG1lJDPnFXjX9HVbO2IMffFu8q4iNCcvcD5b/o7bdj5Fycs/ZZq8M8+qrf
ClTItwKBgQCCrWdYLLP4dMjqJo4EbVjA5OjDj2V7LNWMZwsiW9a8qLebtD2ZoUnz
xL+AH7bWp6muMpx4O9hTpPhvnMVFHffyMZCJruwjIc7o+quAHIGVhjXwv1nGR14G
14sSj4265+JZF6v77TGUKgFUetMtwPdb0TodvTmcBAq5KlCJdL6Y7g==
-----END RSA PRIVATE KEY-----
)EOF";

// ── UART (รับจาก Pi Pico 2W) ─────────────────────
//  Verified working wiring matches SM-PE-01 field test:
//      ESP32 GPIO 18 (RX) ←─────── Pi GPIO16 (TX)
//      ESP32 GPIO 17 (TX) ────────→ Pi GPIO17 (RX)
//      common GND
#define UART_BAUD        2000000   // 2 Mbps
#define UART_RX_PIN      18   // ต่อกับ Pi GPIO16 (TX)
#define UART_TX_PIN      17   // ต่อกับ Pi GPIO17 (RX)
#define UART_BUF_SIZE    8192

// ── LED Status Mapping (forwarded to Pi only; ESP32 does not drive local GPIO) ──
// Pi LED1 = power, Pi LED2 = WiFi link, Pi LED3 = successful SD raw write.
#define LED_POWER        4    // semantic mapping for Pi LED1: power
#define LED_STATUS       5    // semantic mapping for Pi LED2: WiFi connected
#define LED_DATA         19   // Pi LED3: SD raw write success pulse
#define DATA_LED_PULSE_MS 80UL // short pulse so 4 raw chunks/sec still looks like blinking

// ── Packet Protocol ──────────────────────────────
#define PKT_START        0xAA       // (reserved for future custom protocol)
#define PKT_END          0x55       // (reserved for future custom protocol)
#define N_CHANNELS       6          // V1, I1, V2, I2, V3, I3
#define SAMPLES_PER_PKT  100        // (reserved — actual raw chunk is 250 samples)

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
