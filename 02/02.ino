#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Preferences.h>     // NEW: NVS storage for experiment_start_date
#include <SPI.h>
#include <FS.h>
#include <SD.h>
#include <sys/time.h>
#include "config.h"
#include "esp_task_wdt.h"
#include "esp_system.h"

// ============================================================================
//  PowerLens SM-PE-02 — ESP32 side only
//  RAW 1 kHz BINARY CHUNK MODE + SD raw log
//
//  Flow:
//  - ESP32 connects WiFi only for NTP and the WiFi status LED
//  - Sync NTP for daily SD filenames
//  - Request raw electrical data from Pi/EF via UART at target 1 kHz
//  - Store raw int16 samples into binary chunk buffer
//  - Persist raw binary chunk every 250 samples ≈ 250 ms to SD
//  - No Cloud transport in production local-recorder mode
//
//  UART protocol follows Ajarn's ESP32 requester example:
//  ESP32 → Pi:
//      [command_byte][0x0D]
//
//  Pi → ESP32:
//      [0xA5][type][len_lo][len_hi][6 x int16 little-endian][crc_lo][crc_hi][0x0D]
//      total = 19 bytes
//
//  command_byte:
//      bit7 = LED1
//      bit6 = LED2
//      bit5 = LED3
//      bit3 = request data
// ============================================================================

// ── Compile-time defaults / overrides ───────────────────────────────────────
#ifndef UART_BAUD
#define UART_BAUD 2000000
#endif

#ifndef UART_RX_PIN
#define UART_RX_PIN 18
#endif

#ifndef UART_TX_PIN
#define UART_TX_PIN 17
#endif

#ifndef N_CHANNELS
#define N_CHANNELS 6
#endif

#ifndef N_PHASES
#define N_PHASES 3
#endif

#ifndef STATUS_INTERVAL_MS
#define STATUS_INTERVAL_MS 30000UL
#endif

#ifndef WIFI_CONNECT_TIMEOUT_MS
#define WIFI_CONNECT_TIMEOUT_MS 15000UL
#endif

#ifndef NTP_SYNC_TIMEOUT_MS
#define NTP_SYNC_TIMEOUT_MS 8000UL
#endif

#ifndef MQTT_CONNECT_TIMEOUT_MS
#define MQTT_CONNECT_TIMEOUT_MS 25000UL
#endif

#ifndef ENABLE_AWS_IOT
#define ENABLE_AWS_IOT 0
#endif

#ifndef CLOUD_RECONNECT_INTERVAL_MS
#define CLOUD_RECONNECT_INTERVAL_MS 5000UL
#endif

#ifndef FEATURE_PUBLISH_RETRY_MS
#define FEATURE_PUBLISH_RETRY_MS 1000UL
#endif

#ifndef CLOUD_STALL_RESET_MS
#define CLOUD_STALL_RESET_MS 60000UL
#endif

#ifndef FEATURE_STALE_RECOVERY_MS
#define FEATURE_STALE_RECOVERY_MS 15000UL
#endif

#ifndef FEATURE_STALE_RESTART_MS
#define FEATURE_STALE_RESTART_MS 120000UL
#endif

#ifndef PI_METRICS_STALE_MS
#define PI_METRICS_STALE_MS 10000UL
#endif

#ifndef CLOUD_PUBLISH_SLOW_MS
#define CLOUD_PUBLISH_SLOW_MS 3000UL
#endif

#ifndef CLOUD_NET_TIMEOUT_MS
#define CLOUD_NET_TIMEOUT_MS 5000UL
#endif

#ifndef CLOUD_MQTT_SOCKET_TIMEOUT_S
#define CLOUD_MQTT_SOCKET_TIMEOUT_S 5
#endif

#ifndef TEMP_READ_INTERVAL
#define TEMP_READ_INTERVAL 5000UL
#endif

#ifndef DATA_LED_PULSE_MS
#define DATA_LED_PULSE_MS 600UL
#endif

#ifndef MQTT_TOPIC_RAW
#define MQTT_TOPIC_RAW "powerlens/" SITE_ID "/" DEVICE_ID "/raw"
#endif

#ifndef MQTT_TOPIC_FEATURES
#define MQTT_TOPIC_FEATURES "powerlens/" SITE_ID "/" DEVICE_ID "/features"
#endif

// ── Modes ───────────────────────────────────────────────────────────────────
#ifndef ENABLE_PI_ADC_TASK
#define ENABLE_PI_ADC_TASK 1
#endif

#ifndef ENABLE_RAW_BINARY_CHUNK
#define ENABLE_RAW_BINARY_CHUNK 1
#endif

#ifndef ENABLE_RAW_UPLOAD
#define ENABLE_RAW_UPLOAD 0
#endif

#ifndef ENABLE_SD_CARD_LOG
#define ENABLE_SD_CARD_LOG 1
#endif

#ifndef ENABLE_RAW_SD_LOG
#define ENABLE_RAW_SD_LOG 1
#endif

#ifndef ENABLE_FEATURE_SD_LOG
#define ENABLE_FEATURE_SD_LOG 0
#endif

#ifndef ENABLE_AGGREGATE_PUBLISH
#define ENABLE_AGGREGATE_PUBLISH 0
#endif

#ifndef ENABLE_PI_METRICS_PACKET
#define ENABLE_PI_METRICS_PACKET 0
#endif

// ── Sampling / Publish timing ───────────────────────────────────────────────
static const uint32_t SAMPLE_RATE_HZ = 1000UL;
static const uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;
static const uint32_t TELEMETRY_PUBLISH_INTERVAL_MS = 10000UL;
// Feature metrics are requested/published once per second. Raw sampling remains 1 kHz.
static const uint32_t METRICS_REQUEST_INTERVAL_MS = 1000UL;

// Raw chunk = 250 samples × 6 channels × int16 = 3,000 bytes payload.
// Chunk duration follows the current sample rate schedule.
static const uint16_t RAW_CHUNK_SAMPLES = 250;

// 8 buffers × ~3 KB = ~24 KB raw buffer memory.
// This gives the async SD writer about 2 seconds of stall tolerance at 1 kHz
// while leaving enough contiguous heap for AWS IoT TLS handshakes.
#ifndef RAW_BUFFER_COUNT
#define RAW_BUFFER_COUNT 8
#endif

// PubSubClient requires the full publish packet to fit in this buffer.
// Raw upload needs a large packet, but feature-only mode keeps this small so
// TLS has enough contiguous heap for AWS IoT handshakes.
#if ENABLE_RAW_BINARY_CHUNK && ENABLE_RAW_UPLOAD
static const uint16_t MQTT_BUFFER_SIZE_BYTES = 8192;
#else
static const uint16_t MQTT_BUFFER_SIZE_BYTES = 2048;
#endif

// ── SD card logging ─────────────────────────────────────────────────────────
#ifndef SD_CARD_CS_PIN
#define SD_CARD_CS_PIN 26
#endif

#ifndef SD_CARD_SCK_PIN
#define SD_CARD_SCK_PIN 14
#endif

#ifndef SD_CARD_MISO_PIN
#define SD_CARD_MISO_PIN 25
#endif

#ifndef SD_CARD_MOSI_PIN
#define SD_CARD_MOSI_PIN 27
#endif

#ifndef SD_CARD_SPI_FREQ
#define SD_CARD_SPI_FREQ 4000000UL
#endif

#ifndef SD_ROOT_DIR
#define SD_ROOT_DIR "/powerlens"
#endif

#ifndef SD_RAW_FLUSH_INTERVAL_CHUNKS
#define SD_RAW_FLUSH_INTERVAL_CHUNKS 16
#endif

#ifndef SD_FEATURE_FLUSH_INTERVAL_ROWS
#define SD_FEATURE_FLUSH_INTERVAL_ROWS 10
#endif

#ifndef SD_FEATURE_QUEUE_DEPTH
#define SD_FEATURE_QUEUE_DEPTH 16
#endif

#ifndef SD_REMOUNT_INTERVAL_MS
#define SD_REMOUNT_INTERVAL_MS 10000UL
#endif

#ifndef SD_FAILS_BEFORE_COOLDOWN
#define SD_FAILS_BEFORE_COOLDOWN 3
#endif

#ifndef SD_COOLDOWN_MS
#define SD_COOLDOWN_MS 60000UL
#endif

#ifndef ENABLE_COMBINED_SD_LOG
#define ENABLE_COMBINED_SD_LOG 0
#endif

#ifndef SD_COMBINED_FLUSH_INTERVAL_ROWS
#define SD_COMBINED_FLUSH_INTERVAL_ROWS 1000
#endif

#ifndef SD_COMBINED_V_SCALE_PHASE1
#define SD_COMBINED_V_SCALE_PHASE1 0.0785904165665054f
#endif
#ifndef SD_COMBINED_V_SCALE_PHASE2
#define SD_COMBINED_V_SCALE_PHASE2 0.0785904165665054f
#endif
#ifndef SD_COMBINED_V_SCALE_PHASE3
#define SD_COMBINED_V_SCALE_PHASE3 0.0785904165665054f
#endif

#ifndef SD_COMBINED_I_CT_AUTO_ZERO_OFFSET
#define SD_COMBINED_I_CT_AUTO_ZERO_OFFSET 1
#endif

#ifndef SD_COMBINED_I_CT_K
#define SD_COMBINED_I_CT_K 650.7176f
#endif

#ifndef SD_COMBINED_I_CT_EXP
#define SD_COMBINED_I_CT_EXP 1.2f
#endif

#ifndef SD_COMBINED_I_CT_EPSILON_V
#define SD_COMBINED_I_CT_EPSILON_V 0.0001f
#endif

#ifndef SD_COMBINED_I_CT_MAX_A
#define SD_COMBINED_I_CT_MAX_A 300.0f
#endif

#ifndef SD_COMBINED_I_OFFSET_V_PHASE1
#define SD_COMBINED_I_OFFSET_V_PHASE1 0.0f
#endif
#ifndef SD_COMBINED_I_OFFSET_V_PHASE2
#define SD_COMBINED_I_OFFSET_V_PHASE2 0.0f
#endif
#ifndef SD_COMBINED_I_OFFSET_V_PHASE3
#define SD_COMBINED_I_OFFSET_V_PHASE3 0.0f
#endif

// ── Watchdog ────────────────────────────────────────────────────────────────
static const uint32_t WDT_TIMEOUT_S = 30UL;

// ── Pi/EF UART protocol ─────────────────────────────────────────────────────
static const uint8_t EF_END_OF_LINE = 0x0D;
static const uint8_t EF_COMMAND_LENGTH = 2;
static const uint8_t EF_RAW_START_BYTE = 0xA5;
static const uint8_t EF_RAW_TYPE_SAMPLE = 0x01;
static const uint8_t EF_RAW_END_BYTE = 0x5A;
static const uint8_t EF_RAW_PAYLOAD_BYTES = N_CHANNELS * 2;
static const uint8_t EF_RESPONSE_LENGTH = 1 + 1 + 2 + EF_RAW_PAYLOAD_BYTES + 2 + 1;

// For 1 kHz sampling, keep timeout short.
// If Pi answers correctly, response should arrive much faster than 2 ms.
static const uint32_t EF_RESPONSE_TIMEOUT_US = 5000UL;
static const uint32_t EF_METRICS_TIMEOUT_US = 20000UL;
static const uint32_t EF_COMBINED_METRICS_TIMEOUT_US = 4000UL;

// ── Raw binary chunk format ─────────────────────────────────────────────────
struct __attribute__((packed)) RawChunkHeader {
    uint16_t magic;             // 0x504C = 'PL'
    uint8_t schema_ver;         // 2 when utc_epoch_ms is present
    uint8_t header_len;         // sizeof(RawChunkHeader)
    uint32_t seq;               // increasing chunk sequence
    uint64_t timestamp_ms;      // ESP32 millis at chunk start
    uint64_t utc_epoch_ms;      // UTC epoch ms at chunk start; 0 if NTP unavailable
    uint16_t sample_rate_hz;    // 1000
    uint8_t channels;           // 6
    uint16_t sample_count;      // 250
    uint16_t bytes_per_value;   // 2
    uint32_t payload_bytes;     // 3000
    uint32_t checksum32;        // FNV-1a checksum over payload only
};

static const uint16_t RAW_MAGIC = 0x504C;

// ── Pi derived metrics packet format ───────────────────────────────────────
static const uint8_t METRICS_START_BYTE = 0xAA;
static const uint8_t METRICS_TYPE_DERIVED = 0x02;
static const uint8_t METRICS_END_BYTE = 0x55;
static const uint16_t METRICS_PAYLOAD_BYTES = 160;   // was 156; +4 for box_temp_c
static const uint16_t METRICS_PACKET_BYTES = 1 + 1 + 2 + METRICS_PAYLOAD_BYTES + 2 + 1;

struct PiPowerMetrics {
    float Vrms[3];
    float Irms[3];
    float P[3];
    float S[3];
    float PF[3];
    float amp_V[3];
    float amp_I[3];
    float angle_V[3];
    float angle_I[3];
    float phase_diff[3];
    float thd_V[3];
    float thd_I[3];
    float frequency;
    float box_temp_c;             // NEW: Pi enclosure temperature; -127.0 = sensor missing
    uint32_t timestamp_us;
    uint16_t n_samples;
    uint16_t sample_rate_hz;
};


static const size_t RAW_PAYLOAD_BYTES =
    (size_t)RAW_CHUNK_SAMPLES * (size_t)N_CHANNELS * sizeof(int16_t);

static const size_t RAW_CHUNK_TOTAL_BYTES =
    sizeof(RawChunkHeader) + RAW_PAYLOAD_BYTES;

struct RawChunkDesc {
    uint8_t bufferIndex;
    uint32_t seq;
    uint64_t timestampMs;
    uint64_t utcEpochMs;
    uint16_t sampleCount;
    uint16_t sampleRateHz;
    uint32_t payloadBytes;
};

struct FeatureSdRecord {
    PiPowerMetrics metrics;
    uint32_t receivedMs;
    uint64_t utcEpochMs;
    float espTempC;
};

// ── Aggregate structs ───────────────────────────────────────────────────────
struct AggregateState {
    uint32_t samples = 0;
    uint32_t okReads = 0;
    uint32_t failedReads = 0;
    double sum[N_CHANNELS] = {0};
    double sumSq[N_CHANNELS] = {0};
    float minVal[N_CHANNELS] = {0};
    float maxVal[N_CHANNELS] = {0};
    bool initialized = false;
};

struct AggregateSnapshot {
    uint32_t samples = 0;
    uint32_t okReads = 0;
    uint32_t failedReads = 0;
    double sum[N_CHANNELS] = {0};
    double sumSq[N_CHANNELS] = {0};
    float minVal[N_CHANNELS] = {0};
    float maxVal[N_CHANNELS] = {0};
};

// ── Rate Scheduler ─────────────────────────────────────────────────────────
// Sends sample-rate commands to Pi at 18:00 Thailand time daily.
//
// Schedule:
//   Calibrate mode: keep Pi/ADC acquisition at 1 kHz every day.
//   Raw is stored locally on SD; AWS receives lightweight feature payloads only.
//   Day 4+: stays at last
//
// Set-rate packet sent to Pi (7 bytes):
//   [0xFC][period_us LE 4B][CRC8 1B][0xFD]
//
// Architecture note: piSerial is exclusive to adcTask (write @ 1 kHz).
// cloudTask never touches piSerial directly. Instead it enqueues the
// new period into rateChangeQueue, and adcTask dequeues and sends the
// packet between sample requests. This avoids race conditions.
struct RateScheduleEntry {
    int      day_offset;     // 0 = first day
    uint32_t period_us;
    const char* label;
};

static const RateScheduleEntry RATE_SCHEDULE[] = {
    { 0, 1000, "Day 1+: 1 kHz calibrate" },
    { 1, 1000, "Day 1+: 1 kHz calibrate" },
    { 2, 1000, "Day 1+: 1 kHz calibrate" },
};
static const int RATE_SCHEDULE_COUNT = sizeof(RATE_SCHEDULE) / sizeof(RATE_SCHEDULE[0]);

#define RATE_TRANSITION_HOUR     18    // Local Thailand time
#define RATE_TZ_OFFSET_SECONDS   25200 // UTC+7 = 7*3600
#define RATE_SET_PACKET_START    0xFC
#define RATE_SET_PACKET_END      0xFD
#define RATE_NVS_NAMESPACE       "ratesched"
#define RATE_NVS_KEY_START       "exp_start"
#define RATE_NVS_KEY_RESET_TOKEN "rst_token"
#define RATE_TICK_INTERVAL_MS    60000UL  // check schedule once per minute

#ifndef EXPERIMENT_START_RESET_TOKEN
#define EXPERIMENT_START_RESET_TOKEN 0UL
#endif

static QueueHandle_t rateChangeQueue = nullptr;   // payload: uint32_t period_us
static volatile uint32_t currentSamplePeriodUs = SAMPLE_PERIOD_US;  // active period applied by adcTask
static volatile int      currentScheduleDay    = 0;
static time_t            experimentStartEpoch  = 0;     // local-time epoch (UTC + 7h)
static int               lastSentDay           = -1;
static unsigned long     lastRateTickMs        = 0;
static bool              experimentStartLoaded = false;

// Explicit prototypes keep Arduino IDE from generating incorrect declarations
// for complex signatures during .ino preprocessing.
uint32_t fnv1a32(const uint8_t* data, size_t len);
uint16_t crc16Ccitt(const uint8_t* data, uint32_t len);
uint8_t crc8Poly07(const uint8_t* data, size_t len);
void buildRateSetPacket(uint32_t period_us, uint8_t pkt[7]);
time_t getLocalEpochThailand();
uint64_t getUtcEpochMs();
void fillRawChunkHeader(const RawChunkDesc& desc);
bool rawChunkDescIsValid(const RawChunkDesc& desc);
int computeRateScheduleDay(time_t exp_start_local, time_t now_local);
uint32_t periodForScheduleDay(int day);
uint16_t sampleRateHzForPeriodUs(uint32_t period_us);
bool initExperimentStart();
void resetExperimentStart();
bool maybeApplyOneShotExperimentStartReset();
bool requestRateChange(uint32_t period_us);
void rateSchedulerTick();
static float readFloatLE(const uint8_t*& p);
static uint32_t readU32LE(const uint8_t*& p);
static uint16_t readU16LE(const uint8_t*& p);
static void printResetReason();
void setupWatchdog();
void pulseDataLEDNonBlocking();
void updateLEDs();
void refreshLocalLinkState();
void updateTemperature();
bool setupSdCard();
bool mountSdCard(bool verbose);
bool serviceSdCardReconnect();
void markSdCardOffline(const char* reason);
void closeSdFiles();
bool queueFeatureMetricsForSd(const PiPowerMetrics& m, uint32_t receivedMs);
bool writeRawChunkToSd(const RawChunkDesc& desc);
bool writeFeatureRecordToSd(const FeatureSdRecord& rec);
bool writeCombinedSampleChunkToSd(const RawChunkDesc& desc);
bool writeCombinedFeatureRecordToSd(const FeatureSdRecord& rec);
void addSampleToAggregateFloat(const float values[N_CHANNELS]);
void addSampleToAggregateRaw(const int16_t rawValues[N_CHANNELS]);
void addFailedRead();
AggregateSnapshot takeAggregateSnapshotAndReset();
uint8_t buildPiCommandByte(bool requestRaw, bool requestMetrics);
bool requestPiSampleRaw(int16_t outRaw[N_CHANNELS]);
bool requestPiSampleRawAndMetrics(int16_t outRaw[N_CHANNELS],
                                  PiPowerMetrics* outMetrics,
                                  bool* outMetricsReceived);
bool requestPiMetrics(PiPowerMetrics& outMetrics);
void storeLatestPiMetrics(const PiPowerMetrics& m);
bool copyLatestPiMetrics(PiPowerMetrics& m, uint32_t& receivedMs);
void noteCloudProgress(const char* status);
void noteCloudFailure(const char* reason);
#if ENABLE_AWS_IOT
void resetCloudConnection(const char* reason, bool resetWifi);
void publishStatus(const char* connStatus);
bool connectMQTT(uint32_t timeoutMs);
bool publishRawChunk(const RawChunkDesc& desc);
bool publishPiMetrics();
bool publishAggregate(const AggregateSnapshot& snap);
#endif
bool connectWiFi(uint32_t timeoutMs);
bool setupRawQueues();
void adcTask(void* parameter);
void sdWriterTask(void* parameter);
void localServiceTask(void* parameter);
#if ENABLE_AWS_IOT
void cloudTask(void* parameter);
#endif
void setup();
void loop();


// ── Global objects ──────────────────────────────────────────────────────────
#if ENABLE_AWS_IOT
WiFiClientSecure net;
PubSubClient mqtt(net);
#endif
HardwareSerial piSerial(2);

// DS18B20 wiring on ESP32:
//   Use a safe valid GPIO for OneWire's static constructor regardless of config.
//   If TEMP_SENSOR_PIN < 0 (disabled), we never actually init or read — the pin
//   passed here is dummy. This prevents pinMode(-1) warnings during static init.
#if TEMP_SENSOR_PIN < 0
  #define _ONEWIRE_INIT_PIN  16   // dummy; tempSensorEnabled guards all usage
#else
  #define _ONEWIRE_INIT_PIN  TEMP_SENSOR_PIN
#endif
OneWire oneWire(_ONEWIRE_INIT_PIN);
DallasTemperature ds18b20(&oneWire);

// Raw buffers
static uint8_t rawBuffers[RAW_BUFFER_COUNT][RAW_CHUNK_TOTAL_BYTES];
static QueueHandle_t rawFreeQueue = nullptr;
static QueueHandle_t rawFilledQueue = nullptr;
static QueueHandle_t featureSdQueue = nullptr;

static volatile uint32_t rawChunksPublished = 0;
static volatile uint32_t rawChunksFailedPublish = 0;
static volatile uint32_t rawChunksDropped = 0;
static volatile uint32_t rawSamplesOk = 0;
static volatile uint32_t rawSamplesFailed = 0;
static volatile uint32_t rawChunksWrittenSd = 0;
static volatile uint32_t rawChunksFailedSd = 0;
static volatile uint32_t rawFrameCrcFailed = 0;
static volatile uint32_t rawFrameSyncFailed = 0;
static volatile uint32_t featureRowsQueuedSd = 0;
static volatile uint32_t featureRowsDroppedSd = 0;
static volatile uint32_t featureRowsWrittenSd = 0;
static volatile uint32_t featureRowsFailedSd = 0;
static volatile uint32_t combinedSampleRowsWrittenSd = 0;
static volatile uint32_t combinedFeatureRowsWrittenSd = 0;
static volatile uint32_t combinedRowsFailedSd = 0;
static volatile uint32_t sdOfflineEvents = 0;
static volatile uint32_t sdRemountAttempts = 0;
static volatile uint32_t sdRemountOk = 0;
static volatile uint32_t sdCooldownEvents = 0;
static volatile uint32_t sdConsecutiveFailures = 0;
static unsigned long sdCooldownUntilMs = 0;
static char sdLastError[48] = "disabled";
static volatile uint32_t cloudFeaturePublishOk = 0;
static volatile uint32_t cloudFeaturePublishFailed = 0;
static volatile uint32_t cloudPublishSlowCount = 0;
static volatile uint32_t cloudFeatureStaleRecoveries = 0;
static volatile uint32_t cloudFeatureStaleRestarts = 0;
static volatile uint32_t cloudReconnectAttempts = 0;
static volatile uint32_t cloudNetworkResets = 0;
static unsigned long cloudLastProgressMs = 0;
static unsigned long cloudLastReconnectAttemptMs = 0;
static unsigned long cloudLastFeaturePublishOkMs = 0;
static unsigned long cloudLastFeatureRecoveryMs = 0;
static char cloudLastError[48] = "not_started";

#if ENABLE_SD_CARD_LOG
SPIClass sdSPI(VSPI);
static File sdRawFile;
static File sdFeatureFile;
static File sdCombinedFile;
static char sdRawOpenPath[160] = "";
static char sdFeatureOpenPath[160] = "";
static char sdCombinedOpenPath[160] = "";
static char sdBootTag[24] = "";
static uint16_t sdRawChunksSinceFlush = 0;
static uint16_t sdFeatureRowsSinceFlush = 0;
static uint16_t sdCombinedRowsSinceFlush = 0;
static unsigned long lastSdReconnectAttemptMs = 0;
static uint32_t sdMountedSpiFreq = 0;
static bool sdReady = false;
static bool sdWriterTaskStarted = false;
static bool sdSpiStarted = false;
#else
static bool sdReady = false;
static bool sdWriterTaskStarted = false;
#endif

// Latest metrics received from Pi
static PiPowerMetrics latestPiMetrics;
static volatile bool latestPiMetricsValid = false;
static volatile uint32_t latestPiMetricsReceivedMs = 0;
static volatile uint32_t piMetricsOk = 0;
static volatile uint32_t piMetricsFail = 0;
static volatile uint32_t piMetricsConsecutiveFail = 0;
static volatile uint32_t piMetricsStaleRecoveries = 0;
portMUX_TYPE metricsMux = portMUX_INITIALIZER_UNLOCKED;


// ── System state / LED ──────────────────────────────────────────────────────
enum StatusState { CONNECTING, READY, ERROR_STATE };
volatile StatusState sysState = CONNECTING;

static volatile bool dataLedOn = false;
static volatile unsigned long dataLedOffAt = 0;
// Read by adcTask when it forwards LED state. Keeping this as a simple flag
// avoids calling the WiFi driver from the 1 kHz UART request path.
static volatile bool wifiLinkUp = false;

// ── Utility ─────────────────────────────────────────────────────────────────
uint32_t fnv1a32(const uint8_t* data, size_t len) {
    uint32_t hash = 2166136261UL;

    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 16777619UL;
    }

    return hash;
}

uint16_t crc16Ccitt(const uint8_t* data, uint32_t len) {
    uint16_t crc = 0xFFFF;

    for (uint32_t i = 0; i < len; i++) {
        crc ^= ((uint16_t)data[i]) << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000) {
                crc = (uint16_t)((crc << 1) ^ 0x1021);
            } else {
                crc = (uint16_t)(crc << 1);
            }
        }
    }

    return crc;
}

// CRC-8 (poly 0x07, init 0x00) — matches Pi side for set-rate packets.
// Different polynomial from CRC-16 above; do NOT mix the two.
uint8_t crc8Poly07(const uint8_t* data, size_t len) {
    uint8_t crc = 0x00;

    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            if (crc & 0x80) {
                crc = (uint8_t)((crc << 1) ^ 0x07);
            } else {
                crc = (uint8_t)(crc << 1);
            }
        }
    }

    return crc;
}

// ── Rate scheduler helpers ──────────────────────────────────────────────────

// Build the 7-byte set-rate packet for the Pi.
// Caller must provide a 7-byte buffer.
void buildRateSetPacket(uint32_t period_us, uint8_t pkt[7]) {
    pkt[0] = RATE_SET_PACKET_START;
    pkt[1] = (uint8_t)(period_us & 0xFF);
    pkt[2] = (uint8_t)((period_us >> 8) & 0xFF);
    pkt[3] = (uint8_t)((period_us >> 16) & 0xFF);
    pkt[4] = (uint8_t)((period_us >> 24) & 0xFF);
    pkt[5] = crc8Poly07(&pkt[1], 4);
    pkt[6] = RATE_SET_PACKET_END;
}

// Return Thailand local time epoch (UTC + 7h).
// Returns 0 if NTP has not synced yet.
time_t getLocalEpochThailand() {
    time_t utc_now;
    time(&utc_now);
    // sentinel: NTP not synced — time() returns small value (boot epoch)
    if (utc_now < 1700000000) return 0;  // before ~Nov 2023, treat as unsynced
    return utc_now + RATE_TZ_OFFSET_SECONDS;
}

uint64_t getUtcEpochMs() {
    struct timeval tv;
    if (gettimeofday(&tv, nullptr) != 0) {
        return 0;
    }
    if (tv.tv_sec < 1700000000) {
        return 0;
    }
    return ((uint64_t)tv.tv_sec * 1000ULL) + ((uint64_t)tv.tv_usec / 1000ULL);
}

void fillRawChunkHeader(const RawChunkDesc& desc) {
    uint8_t* chunk = rawBuffers[desc.bufferIndex];
    uint8_t* payload = chunk + sizeof(RawChunkHeader);

    RawChunkHeader header;
    header.magic = RAW_MAGIC;
    header.schema_ver = 2;
    header.header_len = sizeof(RawChunkHeader);
    header.seq = desc.seq;
    header.timestamp_ms = desc.timestampMs;
    header.utc_epoch_ms = desc.utcEpochMs;
    header.sample_rate_hz = desc.sampleRateHz;
    header.channels = N_CHANNELS;
    header.sample_count = desc.sampleCount;
    header.bytes_per_value = sizeof(int16_t);
    header.payload_bytes = desc.payloadBytes;
    header.checksum32 = fnv1a32(payload, desc.payloadBytes);

    memcpy(chunk, &header, sizeof(RawChunkHeader));
}

bool rawChunkDescIsValid(const RawChunkDesc& desc) {
    return desc.bufferIndex < RAW_BUFFER_COUNT &&
           desc.sampleCount == RAW_CHUNK_SAMPLES &&
           desc.payloadBytes == RAW_PAYLOAD_BYTES &&
           desc.sampleRateHz > 0;
}

// Compute schedule day index from Thailand local time epoch.
// Day 0 = before any 18:00 boundary; Day k = after kth 18:00 boundary.
int computeRateScheduleDay(time_t exp_start_local, time_t now_local) {
    if (now_local <= exp_start_local) return 0;

    // First boundary = first 18:00 (local) on/after experiment_start
    struct tm t;
    gmtime_r(&exp_start_local, &t);   // we already added +7h, so "UTC fields" = local
    t.tm_hour = RATE_TRANSITION_HOUR;
    t.tm_min  = 0;
    t.tm_sec  = 0;
    // mktime expects tm in local TZ — but our process TZ is unset (UTC).
    // Since exp_start_local already encodes Thailand, we treat it as "naive UTC".
    // Reconstruct via timegm-equivalent (use mktime with TZ=UTC):
    time_t boundary1 = mktime(&t);
    if (boundary1 <= exp_start_local) {
        boundary1 += 86400;
    }

    int day = 0;
    for (int k = 0; k < RATE_SCHEDULE_COUNT - 1; k++) {
        time_t b = boundary1 + (time_t)k * 86400;
        if (now_local >= b) day = k + 1;
        else break;
    }
    if (day >= RATE_SCHEDULE_COUNT) day = RATE_SCHEDULE_COUNT - 1;
    return day;
}

uint32_t periodForScheduleDay(int day) {
    if (day < 0) day = 0;
    if (day >= RATE_SCHEDULE_COUNT) day = RATE_SCHEDULE_COUNT - 1;
    return RATE_SCHEDULE[day].period_us;
}

uint16_t sampleRateHzForPeriodUs(uint32_t period_us) {
    if (period_us == 0) return 0;
    return (uint16_t)(1000000UL / period_us);
}

// Load experiment start date from NVS, or initialize from current time.
// Called once after NTP sync succeeds.
bool initExperimentStart() {
    Preferences prefs;
    if (!prefs.begin(RATE_NVS_NAMESPACE, true)) {
        Serial.println("[RATE] NVS begin (RO) failed — using current time");
    } else {
        uint32_t saved = prefs.getULong(RATE_NVS_KEY_START, 0);
        prefs.end();
        if (saved != 0) {
            experimentStartEpoch = (time_t)saved;
            experimentStartLoaded = true;
            Serial.printf("[RATE] Loaded experiment_start from NVS: %lu (local epoch)\n",
                          (unsigned long)experimentStartEpoch);
            return true;
        }
    }

    // First-ever boot or NVS empty: use today's midnight (local) as start
    time_t now_local = getLocalEpochThailand();
    if (now_local == 0) {
        Serial.println("[RATE] NTP not synced — cannot set experiment_start");
        return false;
    }

    // Round down to local midnight
    struct tm t;
    gmtime_r(&now_local, &t);
    t.tm_hour = 0;
    t.tm_min  = 0;
    t.tm_sec  = 0;
    experimentStartEpoch = mktime(&t);

    Preferences wprefs;
    if (wprefs.begin(RATE_NVS_NAMESPACE, false)) {
        wprefs.putULong(RATE_NVS_KEY_START, (uint32_t)experimentStartEpoch);
        wprefs.end();
        Serial.printf("[RATE] Initialized experiment_start: %lu (local epoch)\n",
                      (unsigned long)experimentStartEpoch);
    }
    experimentStartLoaded = true;
    return true;
}

// Manual override: set experiment_start to today (local midnight)
void resetExperimentStart() {
    time_t now_local = getLocalEpochThailand();
    if (now_local == 0) {
        Serial.println("[RATE] Cannot reset — NTP not synced");
        return;
    }
    struct tm t;
    gmtime_r(&now_local, &t);
    t.tm_hour = 0; t.tm_min = 0; t.tm_sec = 0;
    experimentStartEpoch = mktime(&t);

    Preferences prefs;
    if (prefs.begin(RATE_NVS_NAMESPACE, false)) {
        prefs.putULong(RATE_NVS_KEY_START, (uint32_t)experimentStartEpoch);
        prefs.end();
    }
    lastSentDay = -1;  // force re-send next tick
    Serial.printf("[RATE] RESET experiment_start = %lu\n",
                  (unsigned long)experimentStartEpoch);
}

bool maybeApplyOneShotExperimentStartReset() {
#if EXPERIMENT_START_RESET_TOKEN == 0UL
    return true;
#else
    static bool checked = false;
    if (checked) return true;

    time_t now_local = getLocalEpochThailand();
    if (now_local == 0) {
        return false;
    }

    Preferences prefs;
    if (!prefs.begin(RATE_NVS_NAMESPACE, false)) {
        Serial.println("[RATE] NVS begin (RW) failed — cannot apply one-shot reset");
        return false;
    }

    const uint32_t desiredToken = (uint32_t)EXPERIMENT_START_RESET_TOKEN;
    const uint32_t appliedToken = prefs.getULong(RATE_NVS_KEY_RESET_TOKEN, 0);
    if (appliedToken == desiredToken) {
        prefs.end();
        checked = true;
        return true;
    }

    experimentStartEpoch = now_local;
    prefs.putULong(RATE_NVS_KEY_START, (uint32_t)experimentStartEpoch);
    prefs.putULong(RATE_NVS_KEY_RESET_TOKEN, desiredToken);
    prefs.end();

    experimentStartLoaded = true;
    lastSentDay = -1;

    Serial.printf("[RATE] One-shot reset token %lu applied: experiment_start=%lu (local epoch)\n",
                  (unsigned long)desiredToken,
                  (unsigned long)experimentStartEpoch);

    int initial_day = computeRateScheduleDay(experimentStartEpoch, now_local);
    uint32_t p = periodForScheduleDay(initial_day);
    if (requestRateChange(p)) {
        currentScheduleDay = initial_day;
        lastSentDay = initial_day;
        Serial.printf("[RATE] One-shot Day %d rate queued: period=%lu us (%s)\n",
                      initial_day + 1,
                      (unsigned long)p,
                      RATE_SCHEDULE[initial_day].label);
    } else {
        experimentStartLoaded = false;
        Serial.println("[RATE] One-shot reset applied; rate queue busy, will retry next tick");
    }

    checked = true;
    return true;
#endif
}

// Enqueue rate change for adcTask to send. Returns true on success.
bool requestRateChange(uint32_t period_us) {
    if (rateChangeQueue == nullptr) return false;
    return xQueueSend(rateChangeQueue, &period_us, 0) == pdTRUE;
}

// Called from cloudTask. Self-throttles to once per minute.
void rateSchedulerTick() {
    if (!maybeApplyOneShotExperimentStartReset()) return;

    if (!experimentStartLoaded) {
        // Try to initialize now (only succeeds after NTP)
        initExperimentStart();
        if (!experimentStartLoaded) return;

        // Initial rate send: pick current day and queue immediately
        time_t now_local = getLocalEpochThailand();
        int initial_day = computeRateScheduleDay(experimentStartEpoch, now_local);
        uint32_t p = periodForScheduleDay(initial_day);
        if (requestRateChange(p)) {
            currentScheduleDay = initial_day;
            lastSentDay = initial_day;
            Serial.printf("[RATE] Initial rate queued: Day %d period=%lu us (%s)\n",
                          initial_day + 1, (unsigned long)p,
                          RATE_SCHEDULE[initial_day].label);
        }
        return;
    }

    unsigned long now_ms = millis();
    if ((unsigned long)(now_ms - lastRateTickMs) < RATE_TICK_INTERVAL_MS) return;
    lastRateTickMs = now_ms;

    time_t now_local = getLocalEpochThailand();
    if (now_local == 0) return;

    struct tm t;
    gmtime_r(&now_local, &t);

    // Only consider transitions at/after 18:00 local
    if (t.tm_hour < RATE_TRANSITION_HOUR) return;

    int target_day = computeRateScheduleDay(experimentStartEpoch, now_local);
    if (target_day == lastSentDay) return;

    uint32_t p = periodForScheduleDay(target_day);
    if (requestRateChange(p)) {
        currentScheduleDay = target_day;
        lastSentDay = target_day;
        Serial.printf("[RATE] Transition to Day %d at %02d:%02d local: period=%lu us (%s)\n",
                      target_day + 1, t.tm_hour, t.tm_min,
                      (unsigned long)p,
                      RATE_SCHEDULE[target_day].label);
    } else {
        Serial.println("[RATE] Queue full — will retry next tick");
    }
}

static float readFloatLE(const uint8_t*& p) {
    float v;
    memcpy(&v, p, sizeof(float));
    p += sizeof(float);
    return v;
}

static uint32_t readU32LE(const uint8_t*& p) {
    uint32_t v;
    memcpy(&v, p, sizeof(uint32_t));
    p += sizeof(uint32_t);
    return v;
}

static uint16_t readU16LE(const uint8_t*& p) {
    uint16_t v;
    memcpy(&v, p, sizeof(uint16_t));
    p += sizeof(uint16_t);
    return v;
}

// ── Watchdog Helpers ────────────────────────────────────────────────────────
static void printResetReason() {
    esp_reset_reason_t reason = esp_reset_reason();
    const char* reasonStr = "UNKNOWN";

    switch (reason) {
        case ESP_RST_POWERON:   reasonStr = "POWER_ON";        break;
        case ESP_RST_EXT:       reasonStr = "EXTERNAL_PIN";    break;
        case ESP_RST_SW:        reasonStr = "SW_REBOOT";       break;
        case ESP_RST_PANIC:     reasonStr = "PANIC_or_ASSERT"; break;
        case ESP_RST_INT_WDT:   reasonStr = "INT_WATCHDOG";    break;
        case ESP_RST_TASK_WDT:  reasonStr = "TASK_WATCHDOG";   break;
        case ESP_RST_WDT:       reasonStr = "OTHER_WATCHDOG";  break;
        case ESP_RST_DEEPSLEEP: reasonStr = "WAKE_FROM_DEEPSLEEP"; break;
        case ESP_RST_BROWNOUT:  reasonStr = "BROWNOUT";        break;
        case ESP_RST_SDIO:      reasonStr = "SDIO";            break;
        default: break;
    }

    Serial.printf("[BOOT] Last reset reason: %s (%d)\n", reasonStr, (int)reason);
}

void setupWatchdog() {
    esp_err_t deinitErr = esp_task_wdt_deinit();

    if (deinitErr == ESP_OK) {
        Serial.println("[WDT] Previous TWDT deinitialized");
    } else {
        Serial.printf("[WDT] Previous TWDT deinit skipped/failed: %d\n", deinitErr);
    }

    esp_task_wdt_config_t wdtConfig = {
        .timeout_ms = WDT_TIMEOUT_S * 1000,
        .idle_core_mask = 0,
        .trigger_panic = false
    };

    esp_err_t initErr = esp_task_wdt_init(&wdtConfig);

    if (initErr == ESP_OK) {
        Serial.printf("[WDT] Initialized: %lus\n", (unsigned long)WDT_TIMEOUT_S);
    } else {
        Serial.printf("[WDT] Init failed: %d\n", initErr);
    }
}

// ── LED ─────────────────────────────────────────────────────────────────────
void pulseDataLEDNonBlocking() {
    dataLedOn = true;
    dataLedOffAt = millis() + DATA_LED_PULSE_MS;
}

void updateLEDs() {
    const unsigned long now = millis();
    if (dataLedOn && (long)(now - dataLedOffAt) >= 0) {
        dataLedOn = false;
    }
}

void refreshLocalLinkState() {
    const bool connected = WiFi.status() == WL_CONNECTED;
    const bool wasConnected = wifiLinkUp;
    wifiLinkUp = connected;

    if (connected != wasConnected) {
        Serial.printf("[WiFi] Link %s\n", connected ? "up" : "down");
    }

    // In local-recorder mode, WiFi is the only network dependency. MQTT must
    // not influence the front-panel status because it is deliberately absent.
    sysState = connected ? READY : CONNECTING;
}

// ── DS18B20 async-read state ────────────────────────────────────────────────
static const unsigned long DS18B20_CONVERSION_MS = 750;
float lastTempC = NAN;
unsigned long lastTempRequestMs = 0;
unsigned long lastTempReadyMs = 0;
bool tempConversionPending = false;
bool tempSensorEnabled = true;

void updateTemperature() {
    if (!tempSensorEnabled) return;

    const unsigned long now = millis();

    if (!tempConversionPending && now - lastTempReadyMs >= TEMP_READ_INTERVAL) {
        ds18b20.requestTemperatures();
        lastTempRequestMs = now;
        tempConversionPending = true;
    }

    if (tempConversionPending && now - lastTempRequestMs >= DS18B20_CONVERSION_MS) {
        float t = ds18b20.getTempCByIndex(0);
        lastTempC = (t == DEVICE_DISCONNECTED_C) ? NAN : t;
        tempConversionPending = false;
        lastTempReadyMs = now;
    }
}

// ── SD card logging helpers ─────────────────────────────────────────────────
#if ENABLE_SD_CARD_LOG
static void formatSdDate(uint64_t utcEpochMs, char* dateOut, size_t dateOutLen, bool& synced) {
    synced = utcEpochMs != 0;

    if (!synced) {
        snprintf(dateOut, dateOutLen, "unsynced");
        return;
    }

    time_t localEpoch = (time_t)(utcEpochMs / 1000ULL) + RATE_TZ_OFFSET_SECONDS;
    struct tm t;
    gmtime_r(&localEpoch, &t);
    snprintf(dateOut,
             dateOutLen,
             "%04d-%02d-%02d",
             t.tm_year + 1900,
             t.tm_mon + 1,
             t.tm_mday);
}

static bool buildSdDailyFilePath(char* out,
                                 size_t outLen,
                                 const char* kind,
                                 const char* ext,
                                 uint64_t utcEpochMs) {
    char date[16];
    bool synced = false;
    formatSdDate(utcEpochMs, date, sizeof(date), synced);

    const char* fileTag = synced ? date : sdBootTag;
    int n = snprintf(out,
                     outLen,
                     "%s/%s/%s/%s/%s_%s.%s",
                     SD_ROOT_DIR,
                     kind,
                     DEVICE_ID,
                     date,
                     kind,
                     fileTag,
                     ext);

    return n > 0 && (size_t)n < outLen;
}

static bool ensureSdDirPath(const char* dirPath) {
    if (dirPath == nullptr || dirPath[0] != '/') {
        return false;
    }

    char tmp[160];
    snprintf(tmp, sizeof(tmp), "%s", dirPath);

    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (!SD.exists(tmp) && !SD.mkdir(tmp)) {
                return false;
            }
            *p = '/';
        }
    }

    return SD.exists(tmp) || SD.mkdir(tmp);
}

static bool ensureSdParentDirs(const char* filePath) {
    char dir[160];
    snprintf(dir, sizeof(dir), "%s", filePath);

    char* lastSlash = strrchr(dir, '/');
    if (lastSlash == nullptr) {
        return false;
    }

    *lastSlash = '\0';
    return ensureSdDirPath(dir);
}

static bool openSdAppendFile(File& file,
                             char* currentPath,
                             size_t currentPathLen,
                             const char* wantedPath) {
    if (strncmp(currentPath, wantedPath, currentPathLen) == 0 && file) {
        return true;
    }

    if (file) {
        file.flush();
        file.close();
    }

    if (!ensureSdParentDirs(wantedPath)) {
        Serial.printf("[SD] mkdir failed for %s\n", wantedPath);
        currentPath[0] = '\0';
        return false;
    }

    file = SD.open(wantedPath, FILE_APPEND);
    if (!file) {
        Serial.printf("[SD] open append failed: %s\n", wantedPath);
        currentPath[0] = '\0';
        return false;
    }

    snprintf(currentPath, currentPathLen, "%s", wantedPath);
    return true;
}
#endif

void closeSdFiles() {
#if ENABLE_SD_CARD_LOG
    if (sdRawFile) {
        if (sdReady) {
            sdRawFile.flush();
        }
        sdRawFile.close();
    }
    if (sdFeatureFile) {
        if (sdReady) {
            sdFeatureFile.flush();
        }
        sdFeatureFile.close();
    }
    if (sdCombinedFile) {
        if (sdReady) {
            sdCombinedFile.flush();
        }
        sdCombinedFile.close();
    }
#endif
}

static void resetSdOpenState() {
#if ENABLE_SD_CARD_LOG
    sdRawOpenPath[0] = '\0';
    sdFeatureOpenPath[0] = '\0';
    sdCombinedOpenPath[0] = '\0';
    sdRawChunksSinceFlush = 0;
    sdFeatureRowsSinceFlush = 0;
    sdCombinedRowsSinceFlush = 0;
#endif
}

uint32_t sdCooldownRemainingMs() {
#if ENABLE_SD_CARD_LOG
    const unsigned long untilMs = sdCooldownUntilMs;
    if (untilMs == 0) {
        return 0;
    }

    const unsigned long now = millis();
    if ((int32_t)(untilMs - now) <= 0) {
        sdCooldownUntilMs = 0;
        return 0;
    }

    return (uint32_t)(untilMs - now);
#else
    return 0;
#endif
}

bool sdCooldownActive() {
#if ENABLE_SD_CARD_LOG
    return sdCooldownRemainingMs() > 0;
#else
    return false;
#endif
}

void noteSdSuccess() {
#if ENABLE_SD_CARD_LOG
    sdConsecutiveFailures = 0;
    sdCooldownUntilMs = 0;
#endif
}

void noteSdFailureForCooldown(const char* reason) {
#if ENABLE_SD_CARD_LOG
    sdConsecutiveFailures++;
    if (sdConsecutiveFailures < SD_FAILS_BEFORE_COOLDOWN) {
        return;
    }

    sdCooldownEvents++;
    sdConsecutiveFailures = 0;
    sdCooldownUntilMs = millis() + SD_COOLDOWN_MS;
    sdReady = false;
    closeSdFiles();
    resetSdOpenState();
    SD.end();

    snprintf(sdLastError,
             sizeof(sdLastError),
             "cooldown_%.32s",
             (reason != nullptr && reason[0] != '\0') ? reason : "sd_fail");

    Serial.printf("[SD] Cooldown %lu ms after %s; AWS feature path continues\n",
                  (unsigned long)SD_COOLDOWN_MS,
                  sdLastError);
#else
    (void)reason;
#endif
}

void markSdCardOffline(const char* reason) {
#if ENABLE_SD_CARD_LOG
    if (sdReady) {
        sdOfflineEvents++;
    }

    snprintf(sdLastError,
             sizeof(sdLastError),
             "%s",
             (reason != nullptr && reason[0] != '\0') ? reason : "offline");

    Serial.printf("[SD] Offline: %s\n", sdLastError);

    sdReady = false;
    closeSdFiles();
    resetSdOpenState();
    SD.end();
    noteSdFailureForCooldown(sdLastError);
#else
    (void)reason;
#endif
}

bool mountSdCard(bool verbose) {
#if ENABLE_SD_CARD_LOG
    if (sdCooldownActive()) {
        return false;
    }

    lastSdReconnectAttemptMs = millis();
    sdRemountAttempts++;

    closeSdFiles();
    resetSdOpenState();

    pinMode(SD_CARD_CS_PIN, OUTPUT);
    digitalWrite(SD_CARD_CS_PIN, HIGH);

    if (!sdSpiStarted) {
        sdSPI.begin(SD_CARD_SCK_PIN, SD_CARD_MISO_PIN, SD_CARD_MOSI_PIN, SD_CARD_CS_PIN);
        sdSpiStarted = true;
    }

    const uint32_t tryFreqs[] = { SD_CARD_SPI_FREQ, 4000000UL, 2000000UL, 1000000UL, 400000UL };
    sdReady = false;
    sdMountedSpiFreq = 0;

    for (size_t i = 0; i < sizeof(tryFreqs) / sizeof(tryFreqs[0]); i++) {
        const uint32_t freq = tryFreqs[i];
        bool duplicate = false;
        for (size_t j = 0; j < i; j++) {
            if (tryFreqs[j] == freq) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;

        SD.end();
        digitalWrite(SD_CARD_CS_PIN, HIGH);
        vTaskDelay(pdMS_TO_TICKS(20));

        if (verbose) {
            Serial.printf("[SD] Mount try: spi=%lu Hz (CS=%d SCK=%d MISO=%d MOSI=%d)\n",
                          (unsigned long)freq,
                          SD_CARD_CS_PIN,
                          SD_CARD_SCK_PIN,
                          SD_CARD_MISO_PIN,
                          SD_CARD_MOSI_PIN);
        }

        sdReady = SD.begin(SD_CARD_CS_PIN, sdSPI, freq);
        if (sdReady && SD.cardType() != CARD_NONE) {
            sdMountedSpiFreq = freq;
            break;
        }

        sdReady = false;
    }

    if (!sdReady || SD.cardType() == CARD_NONE) {
        sdReady = false;
        sdMountedSpiFreq = 0;
        snprintf(sdLastError, sizeof(sdLastError), "mount_failed");
        if (verbose) {
            Serial.printf("[SD] Not ready after retries (CS=%d SCK=%d MISO=%d MOSI=%d)\n",
                          SD_CARD_CS_PIN,
                          SD_CARD_SCK_PIN,
                          SD_CARD_MISO_PIN,
                          SD_CARD_MOSI_PIN);
        }
        SD.end();
        noteSdFailureForCooldown("mount_failed");
        return false;
    }

    if (!ensureSdDirPath(SD_ROOT_DIR)) {
        snprintf(sdLastError, sizeof(sdLastError), "mkdir_root_failed");
        if (verbose) {
            Serial.printf("[SD] Root mkdir failed: %s\n", SD_ROOT_DIR);
        }
        sdReady = false;
        sdMountedSpiFreq = 0;
        SD.end();
        noteSdFailureForCooldown("mkdir_root_failed");
        return false;
    }

    snprintf(sdLastError, sizeof(sdLastError), "ready");
    sdRemountOk++;
    noteSdSuccess();

    if (verbose) {
        Serial.printf("[SD] Ready: type=%u size=%llu MB root=%s spi=%lu Hz\n",
                      SD.cardType(),
                      (unsigned long long)(SD.cardSize() / (1024ULL * 1024ULL)),
                      SD_ROOT_DIR,
                      (unsigned long)sdMountedSpiFreq);
    }

    return true;
#else
    (void)verbose;
    return false;
#endif
}

bool serviceSdCardReconnect() {
#if ENABLE_SD_CARD_LOG
    if (sdReady) {
        return true;
    }

    if (sdCooldownActive()) {
        return false;
    }

    const unsigned long now = millis();
    if (now - lastSdReconnectAttemptMs < SD_REMOUNT_INTERVAL_MS) {
        return false;
    }

    return mountSdCard(true);
#else
    return false;
#endif
}

bool setupSdCard() {
#if ENABLE_SD_CARD_LOG
    snprintf(sdBootTag, sizeof(sdBootTag), "boot_%lu", (unsigned long)millis());
    snprintf(sdLastError, sizeof(sdLastError), "not_ready");

    if (featureSdQueue == nullptr) {
        featureSdQueue = xQueueCreate(SD_FEATURE_QUEUE_DEPTH, sizeof(FeatureSdRecord));
    }
    if ((ENABLE_FEATURE_SD_LOG || ENABLE_COMBINED_SD_LOG) && featureSdQueue == nullptr) {
        Serial.println("[SD] Feature queue create FAILED");
    }

    return mountSdCard(true);
#else
    return false;
#endif
}

bool queueFeatureMetricsForSd(const PiPowerMetrics& m, uint32_t receivedMs) {
#if ENABLE_SD_CARD_LOG && (ENABLE_FEATURE_SD_LOG || ENABLE_COMBINED_SD_LOG)
    static uint32_t lastQueuedKey = 0;
    uint32_t key = (m.timestamp_us != 0) ? m.timestamp_us : receivedMs;

    if (key == lastQueuedKey) {
        return true;
    }
    lastQueuedKey = key;

    if (featureSdQueue == nullptr) {
        featureRowsDroppedSd++;
        return false;
    }

    FeatureSdRecord rec;
    memcpy(&rec.metrics, &m, sizeof(PiPowerMetrics));
    rec.receivedMs = receivedMs;
    rec.utcEpochMs = getUtcEpochMs();
    rec.espTempC = lastTempC;

    if (xQueueSend(featureSdQueue, &rec, 0) != pdTRUE) {
        featureRowsDroppedSd++;
        return false;
    }

    featureRowsQueuedSd++;
    return true;
#else
    (void)m;
    (void)receivedMs;
    return true;
#endif
}

bool writeRawChunkToSd(const RawChunkDesc& desc) {
#if ENABLE_SD_CARD_LOG && ENABLE_RAW_SD_LOG
    if (!rawChunkDescIsValid(desc)) {
        rawChunksFailedSd++;
        Serial.printf("[SD] Invalid RAW descriptor: idx=%u samples=%u bytes=%u rate=%u\n",
                      desc.bufferIndex,
                      desc.sampleCount,
                      (unsigned)desc.payloadBytes,
                      desc.sampleRateHz);
        return false;
    }

    if (!sdReady && sdCooldownActive()) {
        rawChunksFailedSd++;
        return false;
    }

    if (!sdReady && !serviceSdCardReconnect()) {
        rawChunksFailedSd++;
        noteSdFailureForCooldown("raw_not_ready");
        return false;
    }

    char path[160];
    if (!buildSdDailyFilePath(path, sizeof(path), "raw", "bin", desc.utcEpochMs)) {
        rawChunksFailedSd++;
        return false;
    }

    if (!openSdAppendFile(sdRawFile, sdRawOpenPath, sizeof(sdRawOpenPath), path)) {
        rawChunksFailedSd++;
        markSdCardOffline("raw_open_failed");
        return false;
    }

    fillRawChunkHeader(desc);

    const uint8_t* chunk = rawBuffers[desc.bufferIndex];
    const size_t totalBytes = sizeof(RawChunkHeader) + desc.payloadBytes;
    const size_t written = sdRawFile.write(chunk, totalBytes);

    if (written != totalBytes) {
        rawChunksFailedSd++;
        Serial.printf("[SD] RAW write short: seq=%u written=%u expected=%u\n",
                      desc.seq,
                      (unsigned)written,
                      (unsigned)totalBytes);
        markSdCardOffline("raw_write_failed");
        if (written == 0 && serviceSdCardReconnect()) {
            if (openSdAppendFile(sdRawFile, sdRawOpenPath, sizeof(sdRawOpenPath), path)) {
                fillRawChunkHeader(desc);
                const size_t retryWritten = sdRawFile.write(chunk, totalBytes);
                if (retryWritten == totalBytes) {
                    rawChunksWrittenSd++;
                    pulseDataLEDNonBlocking();
                    Serial.printf("[SD] RAW write retry OK: seq=%u bytes=%u\n",
                                  desc.seq,
                                  (unsigned)totalBytes);
                    if (++sdRawChunksSinceFlush >= SD_RAW_FLUSH_INTERVAL_CHUNKS) {
                        sdRawFile.flush();
                        sdRawChunksSinceFlush = 0;
                    }
                    return true;
                }
                Serial.printf("[SD] RAW write retry failed: seq=%u written=%u expected=%u\n",
                              desc.seq,
                              (unsigned)retryWritten,
                              (unsigned)totalBytes);
                markSdCardOffline("raw_retry_write_failed");
            }
        }
        return false;
    }

    rawChunksWrittenSd++;
    pulseDataLEDNonBlocking();

    if (++sdRawChunksSinceFlush >= SD_RAW_FLUSH_INTERVAL_CHUNKS) {
        sdRawFile.flush();
        sdRawChunksSinceFlush = 0;
    }

    return true;
#else
    (void)desc;
    return false;
#endif
}

static bool writeFeatureCsvHeaderIfNeeded(File& file) {
#if ENABLE_SD_CARD_LOG && ENABLE_FEATURE_SD_LOG
    if (file.size() > 0) {
        return true;
    }

    file.clearWriteError();
    file.print("schema_ver,device_id,serial_no,site_id,fw_version,utc_epoch_ms,received_ms,pi_timestamp_us,n_samples,sample_rate_hz,frequency,esp_temp_c,pi_box_temp_c");

    for (int i = 1; i <= 3; i++) {
        file.printf(",p%d_Vrms,p%d_Irms,p%d_P,p%d_S,p%d_PF,p%d_amp_V,p%d_amp_I,p%d_angle_V_rad,p%d_angle_I_rad,p%d_phase_diff_rad,p%d_thd_V,p%d_thd_I",
                    i, i, i, i, i, i, i, i, i, i, i, i);
    }

    file.println();
    return file.getWriteError() == 0;
#else
    (void)file;
    return true;
#endif
}

bool writeFeatureRecordToSd(const FeatureSdRecord& rec) {
#if ENABLE_SD_CARD_LOG && ENABLE_FEATURE_SD_LOG
    if (!sdReady && sdCooldownActive()) {
        featureRowsFailedSd++;
        return false;
    }

    if (!sdReady && !serviceSdCardReconnect()) {
        featureRowsFailedSd++;
        noteSdFailureForCooldown("feature_not_ready");
        return false;
    }

    char path[160];
    if (!buildSdDailyFilePath(path, sizeof(path), "features", "csv", rec.utcEpochMs)) {
        featureRowsFailedSd++;
        return false;
    }

    if (!openSdAppendFile(sdFeatureFile, sdFeatureOpenPath, sizeof(sdFeatureOpenPath), path)) {
        featureRowsFailedSd++;
        markSdCardOffline("feature_open_failed");
        return false;
    }

    if (!writeFeatureCsvHeaderIfNeeded(sdFeatureFile)) {
        featureRowsFailedSd++;
        markSdCardOffline("feature_header_failed");
        return false;
    }

    const PiPowerMetrics& m = rec.metrics;
    sdFeatureFile.clearWriteError();
    sdFeatureFile.printf("1,%s,%s,%s,%s,%llu,%lu,%lu,%u,%u,%.3f,",
                         DEVICE_ID,
                         SERIAL_NO,
                         SITE_ID,
                         FW_VERSION,
                         (unsigned long long)rec.utcEpochMs,
                         (unsigned long)rec.receivedMs,
                         (unsigned long)m.timestamp_us,
                         m.n_samples,
                         m.sample_rate_hz,
                         m.frequency);

    if (isnan(rec.espTempC)) {
        sdFeatureFile.print(",");
    } else {
        sdFeatureFile.printf("%.1f,", rec.espTempC);
    }

    if (m.box_temp_c <= -100.0f) {
        sdFeatureFile.print("");
    } else {
        sdFeatureFile.printf("%.1f", m.box_temp_c);
    }

    for (int i = 0; i < 3; i++) {
        sdFeatureFile.printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                             m.Vrms[i],
                             m.Irms[i],
                             m.P[i],
                             m.S[i],
                             m.PF[i],
                             m.amp_V[i],
                             m.amp_I[i],
                             m.angle_V[i],
                             m.angle_I[i],
                             m.phase_diff[i],
                             m.thd_V[i],
                             m.thd_I[i]);
    }

    sdFeatureFile.println();

    if (sdFeatureFile.getWriteError() != 0) {
        featureRowsFailedSd++;
        markSdCardOffline("feature_write_failed");
        return false;
    }

    featureRowsWrittenSd++;

    if (++sdFeatureRowsSinceFlush >= SD_FEATURE_FLUSH_INTERVAL_ROWS) {
        sdFeatureFile.flush();
        sdFeatureRowsSinceFlush = 0;
    }

    return true;
#else
    (void)rec;
    return false;
#endif
}

static const float COMBINED_V_SCALE[N_PHASES] = {
    SD_COMBINED_V_SCALE_PHASE1,
    SD_COMBINED_V_SCALE_PHASE2,
    SD_COMBINED_V_SCALE_PHASE3
};

static const float COMBINED_I_OFFSET_V[N_PHASES] = {
    SD_COMBINED_I_OFFSET_V_PHASE1,
    SD_COMBINED_I_OFFSET_V_PHASE2,
    SD_COMBINED_I_OFFSET_V_PHASE3
};

static int16_t readRawValueFromChunk(const RawChunkDesc& desc, uint16_t sampleIndex, uint8_t channel) {
    int16_t value = 0;
    const uint8_t* payload = rawBuffers[desc.bufferIndex] + sizeof(RawChunkHeader);
    const size_t offset =
        ((size_t)sampleIndex * (size_t)N_CHANNELS + (size_t)channel) * sizeof(int16_t);

    if (offset + sizeof(int16_t) <= desc.payloadBytes) {
        memcpy(&value, payload + offset, sizeof(int16_t));
    }

    return value;
}

static float combinedCtVoltageToCurrentA(float vCt, float offsetV) {
    if (!isfinite(vCt) || !isfinite(offsetV)) return 0.0f;

    float x = vCt - offsetV;
    if (!isfinite(x) || fabsf(x) < SD_COMBINED_I_CT_EPSILON_V) {
        return 0.0f;
    }

    float mag = SD_COMBINED_I_CT_K * powf(fabsf(x), SD_COMBINED_I_CT_EXP);
    if (!isfinite(mag)) return 0.0f;

    if (mag > SD_COMBINED_I_CT_MAX_A) {
        mag = SD_COMBINED_I_CT_MAX_A;
    }

    return (x < 0.0f) ? -mag : mag;
}

static void computeCombinedCurrentOffsets(const RawChunkDesc& desc, float offsetsOut[N_PHASES]) {
    for (int p = 0; p < N_PHASES; p++) {
        offsetsOut[p] = COMBINED_I_OFFSET_V[p];
    }

#if SD_COMBINED_I_CT_AUTO_ZERO_OFFSET
    if (desc.sampleCount == 0) return;

    float sums[N_PHASES] = {0.0f, 0.0f, 0.0f};
    for (uint16_t s = 0; s < desc.sampleCount; s++) {
        sums[0] += (float)readRawValueFromChunk(desc, s, 1) / 5000.0f;
        sums[1] += (float)readRawValueFromChunk(desc, s, 3) / 5000.0f;
        sums[2] += (float)readRawValueFromChunk(desc, s, 5) / 5000.0f;
    }

    for (int p = 0; p < N_PHASES; p++) {
        offsetsOut[p] = sums[p] / (float)desc.sampleCount;
    }
#endif
}

static void writeCombinedCsvHeaderIfNeeded(File& file) {
#if ENABLE_SD_CARD_LOG && ENABLE_COMBINED_SD_LOG
    if (file.size() > 0) {
        return;
    }

    file.print("row_type,schema_ver,device_id,serial_no,site_id,fw_version,utc_epoch_ms,millis_ms,seq,sample_index,sample_offset_us,sample_rate_hz,v1_v,i1_a,v2_v,i2_a,v3_v,i3_a,i1_offset_v,i2_offset_v,i3_offset_v,feature_received_ms,pi_timestamp_us,n_samples,frequency_hz,esp_temp_c,pi_box_temp_c");

    for (int i = 1; i <= 3; i++) {
        file.printf(",p%d_Vrms,p%d_Irms,p%d_P,p%d_S,p%d_PF,p%d_amp_V,p%d_amp_I,p%d_angle_V_rad,p%d_angle_I_rad,p%d_phase_diff_rad,p%d_thd_V,p%d_thd_I",
                    i, i, i, i, i, i, i, i, i, i, i, i);
    }

    file.println();
#else
    (void)file;
#endif
}

static bool openCombinedSdFile(uint64_t utcEpochMs) {
#if ENABLE_SD_CARD_LOG && ENABLE_COMBINED_SD_LOG
    if (!sdReady && sdCooldownActive()) {
        combinedRowsFailedSd++;
        return false;
    }

    if (!sdReady && !serviceSdCardReconnect()) {
        combinedRowsFailedSd++;
        noteSdFailureForCooldown("combined_not_ready");
        return false;
    }

    char path[160];
    if (!buildSdDailyFilePath(path, sizeof(path), "combined", "csv", utcEpochMs)) {
        combinedRowsFailedSd++;
        return false;
    }

    if (!openSdAppendFile(sdCombinedFile, sdCombinedOpenPath, sizeof(sdCombinedOpenPath), path)) {
        combinedRowsFailedSd++;
        markSdCardOffline("combined_open_failed");
        return false;
    }

    writeCombinedCsvHeaderIfNeeded(sdCombinedFile);
    return true;
#else
    (void)utcEpochMs;
    return false;
#endif
}

static void flushCombinedIfNeeded(uint16_t rowsAdded) {
#if ENABLE_SD_CARD_LOG && ENABLE_COMBINED_SD_LOG
    sdCombinedRowsSinceFlush += rowsAdded;
    if (sdCombinedRowsSinceFlush >= SD_COMBINED_FLUSH_INTERVAL_ROWS) {
        sdCombinedFile.flush();
        sdCombinedRowsSinceFlush = 0;
    }
#else
    (void)rowsAdded;
#endif
}

bool writeCombinedSampleChunkToSd(const RawChunkDesc& desc) {
#if ENABLE_SD_CARD_LOG && ENABLE_COMBINED_SD_LOG
    if (!openCombinedSdFile(desc.utcEpochMs)) {
        return false;
    }

    float iOffsets[N_PHASES];
    computeCombinedCurrentOffsets(desc, iOffsets);

    sdCombinedFile.clearWriteError();

    for (uint16_t s = 0; s < desc.sampleCount; s++) {
        uint64_t sampleOffsetUs = 0;
        if (desc.sampleRateHz > 0) {
            sampleOffsetUs = ((uint64_t)s * 1000000ULL) / (uint64_t)desc.sampleRateHz;
        }

        uint64_t sampleUtcMs = 0;
        if (desc.utcEpochMs != 0) {
            sampleUtcMs = desc.utcEpochMs + (sampleOffsetUs / 1000ULL);
        }
        uint64_t sampleMillisMs = desc.timestampMs + (sampleOffsetUs / 1000ULL);

        const float v1 = (float)readRawValueFromChunk(desc, s, 0) * COMBINED_V_SCALE[0];
        const float v2 = (float)readRawValueFromChunk(desc, s, 2) * COMBINED_V_SCALE[1];
        const float v3 = (float)readRawValueFromChunk(desc, s, 4) * COMBINED_V_SCALE[2];

        const float i1v = (float)readRawValueFromChunk(desc, s, 1) / 5000.0f;
        const float i2v = (float)readRawValueFromChunk(desc, s, 3) / 5000.0f;
        const float i3v = (float)readRawValueFromChunk(desc, s, 5) / 5000.0f;

        const float i1 = combinedCtVoltageToCurrentA(i1v, iOffsets[0]);
        const float i2 = combinedCtVoltageToCurrentA(i2v, iOffsets[1]);
        const float i3 = combinedCtVoltageToCurrentA(i3v, iOffsets[2]);

        sdCombinedFile.printf("sample,1,%s,%s,%s,%s,%llu,%llu,%u,%u,%llu,%u,%.5f,%.6f,%.5f,%.6f,%.5f,%.6f,%.6f,%.6f,%.6f",
                              DEVICE_ID,
                              SERIAL_NO,
                              SITE_ID,
                              FW_VERSION,
                              (unsigned long long)sampleUtcMs,
                              (unsigned long long)sampleMillisMs,
                              desc.seq,
                              s,
                              (unsigned long long)sampleOffsetUs,
                              desc.sampleRateHz,
                              v1,
                              i1,
                              v2,
                              i2,
                              v3,
                              i3,
                              iOffsets[0],
                              iOffsets[1],
                              iOffsets[2]);

        for (int empty = 0; empty < 42; empty++) {
            sdCombinedFile.print(",");
        }
        sdCombinedFile.println();
    }

    if (sdCombinedFile.getWriteError() != 0) {
        combinedRowsFailedSd++;
        markSdCardOffline("combined_sample_write_failed");
        return false;
    }

    combinedSampleRowsWrittenSd += desc.sampleCount;
    flushCombinedIfNeeded(desc.sampleCount);
    return true;
#else
    (void)desc;
    return true;
#endif
}

static void printCombinedFloatOrEmpty(File& file, float value, uint8_t decimals) {
    if (isfinite(value)) {
        file.printf("%.*f", decimals, value);
    }
}

bool writeCombinedFeatureRecordToSd(const FeatureSdRecord& rec) {
#if ENABLE_SD_CARD_LOG && ENABLE_COMBINED_SD_LOG
    if (!openCombinedSdFile(rec.utcEpochMs)) {
        return false;
    }

    const PiPowerMetrics& m = rec.metrics;
    sdCombinedFile.clearWriteError();

    sdCombinedFile.printf("feature,1,%s,%s,%s,%s,%llu,%lu",
                          DEVICE_ID,
                          SERIAL_NO,
                          SITE_ID,
                          FW_VERSION,
                          (unsigned long long)rec.utcEpochMs,
                          (unsigned long)rec.receivedMs);

    sdCombinedFile.print(",,,,");
    sdCombinedFile.print(m.sample_rate_hz);
    sdCombinedFile.print(",,,,,,,,,,");
    sdCombinedFile.printf("%lu,%lu,%u,%.3f,",
                          (unsigned long)rec.receivedMs,
                          (unsigned long)m.timestamp_us,
                          m.n_samples,
                          m.frequency);

    if (isfinite(rec.espTempC)) {
        sdCombinedFile.printf("%.1f", rec.espTempC);
    }
    sdCombinedFile.print(",");

    if (m.box_temp_c > -100.0f && isfinite(m.box_temp_c)) {
        sdCombinedFile.printf("%.1f", m.box_temp_c);
    }

    for (int i = 0; i < 3; i++) {
        sdCombinedFile.printf(",%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                              m.Vrms[i],
                              m.Irms[i],
                              m.P[i],
                              m.S[i],
                              m.PF[i],
                              m.amp_V[i],
                              m.amp_I[i],
                              m.angle_V[i],
                              m.angle_I[i],
                              m.phase_diff[i],
                              m.thd_V[i],
                              m.thd_I[i]);
    }

    sdCombinedFile.println();

    if (sdCombinedFile.getWriteError() != 0) {
        combinedRowsFailedSd++;
        markSdCardOffline("combined_feature_write_failed");
        return false;
    }

    combinedFeatureRowsWrittenSd++;
    flushCombinedIfNeeded(1);
    return true;
#else
    (void)rec;
    return true;
#endif
}

// ── Aggregate storage shared between tasks ──────────────────────────────────
AggregateState agg;
portMUX_TYPE aggMux = portMUX_INITIALIZER_UNLOCKED;

void addSampleToAggregateFloat(const float values[N_CHANNELS]) {
    portENTER_CRITICAL(&aggMux);

    if (!agg.initialized) {
        for (int ch = 0; ch < N_CHANNELS; ch++) {
            agg.minVal[ch] = values[ch];
            agg.maxVal[ch] = values[ch];
        }
        agg.initialized = true;
    }

    agg.samples++;
    agg.okReads++;

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const float v = values[ch];

        agg.sum[ch] += v;
        agg.sumSq[ch] += (double)v * (double)v;

        if (v < agg.minVal[ch]) agg.minVal[ch] = v;
        if (v > agg.maxVal[ch]) agg.maxVal[ch] = v;
    }

    portEXIT_CRITICAL(&aggMux);
}

void addSampleToAggregateRaw(const int16_t rawValues[N_CHANNELS]) {
    float values[N_CHANNELS];

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        values[ch] = ((float)rawValues[ch]) / 5.0f;
    }

    addSampleToAggregateFloat(values);
}

void addFailedRead() {
    portENTER_CRITICAL(&aggMux);
    agg.failedReads++;
    portEXIT_CRITICAL(&aggMux);
}

AggregateSnapshot takeAggregateSnapshotAndReset() {
    AggregateSnapshot snap;

    portENTER_CRITICAL(&aggMux);

    snap.samples = agg.samples;
    snap.okReads = agg.okReads;
    snap.failedReads = agg.failedReads;

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        snap.sum[ch] = agg.sum[ch];
        snap.sumSq[ch] = agg.sumSq[ch];
        snap.minVal[ch] = agg.minVal[ch];
        snap.maxVal[ch] = agg.maxVal[ch];

        agg.sum[ch] = 0;
        agg.sumSq[ch] = 0;
        agg.minVal[ch] = 0;
        agg.maxVal[ch] = 0;
    }

    agg.samples = 0;
    agg.okReads = 0;
    agg.failedReads = 0;
    agg.initialized = false;

    portEXIT_CRITICAL(&aggMux);

    return snap;
}

// ── Pi/EF UART request-response ─────────────────────────────────────────────
uint8_t buildPiCommandByte(bool requestRaw, bool requestMetrics) {
    /*
      Protocol ตามไฟล์ Pi ใหม่

      ESP32 ส่ง command 1 byte + 0x0D ไปหา Pi

      bit7 = LED1
      bit6 = LED2
      bit5 = LED3
      bit3 = request RAW snapshot
      bit2 = request DERIVED metrics packet
    */

    const uint8_t led1 = 1;
#if ENABLE_AWS_IOT
    const uint8_t led2 = mqtt.connected() ? 1 : 0;
#else
    const uint8_t led2 = wifiLinkUp ? 1 : 0;
#endif
    const uint8_t led3 = dataLedOn ? 1 : 0;

    uint8_t command = 0x00;

    command |= ((led1 & 0x01) << 7);
    command |= ((led2 & 0x01) << 6);
    command |= ((led3 & 0x01) << 5);

    if (requestRaw) {
        command |= (0x01 << 3);
    }

    if (requestMetrics) {
        command |= (0x01 << 2);
    }

    return command;
}

bool requestPiSampleRaw(int16_t outRaw[N_CHANNELS]) {
    bool metricsReceived = false;
    return requestPiSampleRawAndMetrics(outRaw, nullptr, &metricsReceived);
}



static void clearPiSerialRx() {
    while (piSerial.available()) {
        piSerial.read();
    }
}

static bool sendPiCommand(bool requestRaw, bool requestMetrics) {
    const uint8_t commandBuffer[EF_COMMAND_LENGTH] = {
        buildPiCommandByte(requestRaw, requestMetrics),
        EF_END_OF_LINE
    };

    const size_t written = piSerial.write(commandBuffer, EF_COMMAND_LENGTH);
    piSerial.flush();
    return written == EF_COMMAND_LENGTH;
}

static bool readPiRawFrame(int16_t outRaw[N_CHANNELS]) {
    uint8_t frame[EF_RESPONSE_LENGTH];
    uint8_t idx = 0;
    const uint32_t startUs = micros();

    while (idx < EF_RESPONSE_LENGTH &&
           (uint32_t)(micros() - startUs) < EF_RESPONSE_TIMEOUT_US) {
        int b = piSerial.read();

        if (b >= 0) {
            if (idx == 0 && (uint8_t)b != EF_RAW_START_BYTE) {
                continue;
            }
            frame[idx++] = (uint8_t)b;
        } else {
            delayMicroseconds(30);
        }
    }

    if (idx != EF_RESPONSE_LENGTH) {
        rawFrameSyncFailed++;
        return false;
    }

    if (frame[0] != EF_RAW_START_BYTE ||
        frame[1] != EF_RAW_TYPE_SAMPLE ||
        frame[EF_RESPONSE_LENGTH - 1] != EF_RAW_END_BYTE) {
        rawFrameSyncFailed++;
        return false;
    }

    const uint16_t crcRx =
        (uint16_t)frame[4 + EF_RAW_PAYLOAD_BYTES] |
        ((uint16_t)frame[4 + EF_RAW_PAYLOAD_BYTES + 1] << 8);
    const uint16_t crcCalc = crc16Ccitt(&frame[2], 2 + EF_RAW_PAYLOAD_BYTES);
    if (crcRx != crcCalc) {
        rawFrameCrcFailed++;
        return false;
    }

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        const uint8_t payloadOffset = 4 + (2 * ch);
        int16_t raw = (int16_t)(
            ((uint16_t)frame[payloadOffset]) |
            ((uint16_t)frame[payloadOffset + 1] << 8)
        );
        outRaw[ch] = raw;
    }

    return true;
}

static bool readPiMetricsPacket(PiPowerMetrics& outMetrics, uint32_t timeoutUs) {
    uint8_t packet[METRICS_PACKET_BYTES];
    uint16_t idx = 0;
    const uint32_t startUs = micros();

    while (idx < METRICS_PACKET_BYTES &&
           (uint32_t)(micros() - startUs) < timeoutUs) {
        int b = piSerial.read();

        if (b >= 0) {
            packet[idx++] = (uint8_t)b;
        } else {
            delayMicroseconds(30);
        }
    }

    if (idx != METRICS_PACKET_BYTES) {
        return false;
    }

    if (packet[0] != METRICS_START_BYTE ||
        packet[1] != METRICS_TYPE_DERIVED ||
        packet[METRICS_PACKET_BYTES - 1] != METRICS_END_BYTE) {
        return false;
    }

    uint16_t len = (uint16_t)packet[2] | ((uint16_t)packet[3] << 8);
    if (len != METRICS_PAYLOAD_BYTES) {
        return false;
    }

    const uint8_t* p = &packet[4];
    uint16_t crcRx = (uint16_t)packet[4 + len] | ((uint16_t)packet[4 + len + 1] << 8);
    uint16_t crcCalc = crc16Ccitt(p, len);

    if (crcRx != crcCalc) {
        Serial.printf("[METRICS] CRC mismatch rx=0x%04X calc=0x%04X\n", crcRx, crcCalc);
        return false;
    }

    for (int i = 0; i < 3; i++) outMetrics.Vrms[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.Irms[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.P[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.S[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.PF[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.amp_V[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.amp_I[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.angle_V[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.angle_I[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.phase_diff[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.thd_V[i] = readFloatLE(p);
    for (int i = 0; i < 3; i++) outMetrics.thd_I[i] = readFloatLE(p);
    outMetrics.frequency = readFloatLE(p);
    outMetrics.box_temp_c = readFloatLE(p);
    outMetrics.timestamp_us = readU32LE(p);
    outMetrics.n_samples = readU16LE(p);
    outMetrics.sample_rate_hz = readU16LE(p);

    return true;
}

bool requestPiSampleRawAndMetrics(int16_t outRaw[N_CHANNELS],
                                  PiPowerMetrics* outMetrics,
                                  bool* outMetricsReceived) {
    if (outMetricsReceived != nullptr) {
        *outMetricsReceived = false;
    }

    clearPiSerialRx();

    if (!sendPiCommand(true, outMetrics != nullptr)) {
        return false;
    }

    if (!readPiRawFrame(outRaw)) {
        clearPiSerialRx();
        return false;
    }

    if (outMetrics != nullptr) {
        PiPowerMetrics metrics;
        if (readPiMetricsPacket(metrics, EF_COMBINED_METRICS_TIMEOUT_US)) {
            *outMetrics = metrics;
            if (outMetricsReceived != nullptr) {
                *outMetricsReceived = true;
            }
        }
    }

    clearPiSerialRx();
    return true;
}

bool requestPiMetrics(PiPowerMetrics& outMetrics) {
#if ENABLE_PI_METRICS_PACKET
    clearPiSerialRx();

    if (!sendPiCommand(false, true)) {
        return false;
    }

    bool ok = readPiMetricsPacket(outMetrics, EF_METRICS_TIMEOUT_US);
    clearPiSerialRx();
    return ok;
#else
    (void)outMetrics;
    return false;
#endif
}

void storeLatestPiMetrics(const PiPowerMetrics& m) {
    portENTER_CRITICAL(&metricsMux);
    memcpy(&latestPiMetrics, &m, sizeof(PiPowerMetrics));
    latestPiMetricsValid = true;
    latestPiMetricsReceivedMs = millis();
    piMetricsOk++;
    piMetricsConsecutiveFail = 0;
    portEXIT_CRITICAL(&metricsMux);
}

void invalidateLatestPiMetrics() {
    portENTER_CRITICAL(&metricsMux);
    latestPiMetricsValid = false;
    latestPiMetricsReceivedMs = 0;
    portEXIT_CRITICAL(&metricsMux);
}

uint32_t latestPiMetricsAgeMs(uint32_t nowMs) {
    uint32_t receivedMs = 0;
    bool valid = false;

    portENTER_CRITICAL(&metricsMux);
    valid = latestPiMetricsValid;
    receivedMs = latestPiMetricsReceivedMs;
    portEXIT_CRITICAL(&metricsMux);

    if (!valid || receivedMs == 0) {
        return UINT32_MAX;
    }
    return nowMs - receivedMs;
}

bool copyLatestPiMetrics(PiPowerMetrics& m, uint32_t& receivedMs) {
    bool valid;

    portENTER_CRITICAL(&metricsMux);
    valid = latestPiMetricsValid;
    if (valid) {
        memcpy(&m, &latestPiMetrics, sizeof(PiPowerMetrics));
        receivedMs = latestPiMetricsReceivedMs;
    }
    portEXIT_CRITICAL(&metricsMux);

    return valid;
}

// ── WiFi / MQTT ─────────────────────────────────────────────────────────────
void noteCloudProgress(const char* status) {
    cloudLastProgressMs = millis();
    snprintf(cloudLastError,
             sizeof(cloudLastError),
             "%s",
             (status != nullptr && status[0] != '\0') ? status : "ok");
}

void noteCloudFailure(const char* reason) {
    snprintf(cloudLastError,
             sizeof(cloudLastError),
             "%s",
             (reason != nullptr && reason[0] != '\0') ? reason : "cloud_fail");
}

#if ENABLE_AWS_IOT
void resetCloudConnection(const char* reason, bool resetWifi) {
    cloudNetworkResets++;
    noteCloudFailure(reason);
    Serial.printf("[CLOUD] Reset network stack: %s reset_wifi=%d\n",
                  cloudLastError,
                  resetWifi ? 1 : 0);

    mqtt.disconnect();
    net.stop();

    if (resetWifi) {
        WiFi.disconnect(false, false);
        WiFi.mode(WIFI_OFF);
        vTaskDelay(pdMS_TO_TICKS(200));
    }
}
#endif

bool connectWiFi(uint32_t timeoutMs) {
    if (WiFi.status() == WL_CONNECTED) {
        refreshLocalLinkState();
        return true;
    }

    sysState = CONNECTING;
    cloudReconnectAttempts++;

    Serial.printf("[WiFi] Connecting to %s\n", WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.persistent(false);
    WiFi.setSleep(false);
    WiFi.setAutoReconnect(true);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    const unsigned long startMs = millis();
    while (WiFi.status() != WL_CONNECTED &&
           (uint32_t)(millis() - startMs) < timeoutMs) {
        esp_task_wdt_reset();
        updateLEDs();
        vTaskDelay(pdMS_TO_TICKS(100));
    }

    if (WiFi.status() != WL_CONNECTED) {
        refreshLocalLinkState();
        Serial.printf("[WiFi] Connect timeout after %lu ms\n", (unsigned long)timeoutMs);
        noteCloudFailure("wifi_timeout");
        return false;
    }

    Serial.printf("[WiFi] Connected: %s\n", WiFi.localIP().toString().c_str());
    refreshLocalLinkState();
    noteCloudProgress("wifi_connected");

    Serial.println("[NTP] Syncing time...");

    configTime(0, 0, "pool.ntp.org", "time.nist.gov", "time.google.com");

    struct tm timeinfo;
    unsigned long ntpStartMs = millis();
    bool ntpOk = false;

    while ((uint32_t)(millis() - ntpStartMs) < NTP_SYNC_TIMEOUT_MS) {
        esp_task_wdt_reset();
        if (getLocalTime(&timeinfo, 500)) {
            ntpOk = true;
            break;
        }
        Serial.println("[NTP] Waiting for time sync...");
        vTaskDelay(pdMS_TO_TICKS(250));
    }

    if (ntpOk) {
        Serial.printf("[NTP] Time synced: %04d-%02d-%02d %02d:%02d:%02d UTC\n",
                      timeinfo.tm_year + 1900,
                      timeinfo.tm_mon + 1,
                      timeinfo.tm_mday,
                      timeinfo.tm_hour,
                      timeinfo.tm_min,
                      timeinfo.tm_sec);
        noteCloudProgress("ntp_synced");
    } else {
        Serial.println("[NTP] WARNING: Time sync failed — new SD files use unsynced timestamps");
        noteCloudFailure("ntp_timeout");
    }

    return true;
}

// ── Cloud-only transport and payloads ───────────────────────────────────────
#if ENABLE_AWS_IOT
// Pre-allocated to avoid heap churn in the optional AWS build.
static StaticJsonDocument<4096> statusDoc;
static StaticJsonDocument<4096> telemetryDoc;
static StaticJsonDocument<4096> metricsDoc;
static char statusPayload[4096];
static char telemetryPayload[4096];
static char metricsPayload[4096];

void publishStatus(const char* connStatus) {
    statusDoc.clear();

    statusDoc["schema_ver"] = 1;
    statusDoc["device_id"] = DEVICE_ID;
    statusDoc["serial_no"] = SERIAL_NO;
    statusDoc["site_id"] = SITE_ID;
    statusDoc["fw_version"] = FW_VERSION;
    statusDoc["status"] = connStatus;
    const uint64_t statusUtcEpochMs = getUtcEpochMs();
    if (statusUtcEpochMs != 0) {
        statusDoc["utc_epoch_ms"] = statusUtcEpochMs;
    }
    statusDoc["ip"] = WiFi.localIP().toString();
    statusDoc["rssi"] = WiFi.RSSI();
    statusDoc["free_heap"] = ESP.getFreeHeap();
    statusDoc["uptime_s"] = (uint32_t)(millis() / 1000UL);

    const char* rawMode = "off";
    if (ENABLE_RAW_BINARY_CHUNK && ENABLE_RAW_SD_LOG && ENABLE_COMBINED_SD_LOG) {
        rawMode = ENABLE_RAW_UPLOAD ? "sd_raw_bin_combined_csv_and_mqtt_binary_chunk_250samp" : "sd_raw_bin_and_combined_csv_250samp";
    } else if (ENABLE_RAW_BINARY_CHUNK && ENABLE_RAW_SD_LOG) {
        rawMode = ENABLE_RAW_UPLOAD ? "sd_and_mqtt_binary_chunk_250samp" : "sd_binary_chunk_250samp";
    } else if (ENABLE_RAW_BINARY_CHUNK && ENABLE_COMBINED_SD_LOG) {
        rawMode = "sd_combined_csv_250samp";
    } else if (ENABLE_RAW_BINARY_CHUNK && ENABLE_RAW_UPLOAD) {
        rawMode = "mqtt_binary_chunk_250samp";
    } else if (ENABLE_RAW_BINARY_CHUNK) {
        rawMode = "feature_only_no_raw_upload";
    }

    statusDoc["raw_mode"] = rawMode;
    statusDoc["raw_chunk_samples"] = RAW_CHUNK_SAMPLES;
    statusDoc["raw_payload_bytes"] = RAW_PAYLOAD_BYTES;
    statusDoc["raw_chunks_published"] = (uint32_t)rawChunksPublished;
    statusDoc["raw_chunks_failed"] = (uint32_t)rawChunksFailedPublish;
    statusDoc["raw_chunks_dropped"] = (uint32_t)rawChunksDropped;
    statusDoc["raw_samples_ok"] = (uint32_t)rawSamplesOk;
    statusDoc["raw_samples_failed"] = (uint32_t)rawSamplesFailed;
    statusDoc["raw_frame_crc_failed"] = (uint32_t)rawFrameCrcFailed;
    statusDoc["raw_frame_sync_failed"] = (uint32_t)rawFrameSyncFailed;
    statusDoc["sd_enabled"] = ENABLE_SD_CARD_LOG ? true : false;
    statusDoc["sd_ready"] = sdReady ? true : false;
    statusDoc["sd_last_error"] = sdLastError;
    statusDoc["sd_offline_events"] = (uint32_t)sdOfflineEvents;
    statusDoc["sd_remount_attempts"] = (uint32_t)sdRemountAttempts;
    statusDoc["sd_remount_ok"] = (uint32_t)sdRemountOk;
    statusDoc["sd_cooldown_events"] = (uint32_t)sdCooldownEvents;
    statusDoc["sd_cooldown_ms"] = sdCooldownRemainingMs();
    statusDoc["sd_consecutive_failures"] = (uint32_t)sdConsecutiveFailures;
    statusDoc["sd_raw_chunks_written"] = (uint32_t)rawChunksWrittenSd;
    statusDoc["sd_raw_chunks_failed"] = (uint32_t)rawChunksFailedSd;
    statusDoc["sd_feature_rows_queued"] = (uint32_t)featureRowsQueuedSd;
    statusDoc["sd_feature_rows_dropped"] = (uint32_t)featureRowsDroppedSd;
    statusDoc["sd_feature_rows_written"] = (uint32_t)featureRowsWrittenSd;
    statusDoc["sd_feature_rows_failed"] = (uint32_t)featureRowsFailedSd;
    statusDoc["sd_combined_sample_rows_written"] = (uint32_t)combinedSampleRowsWrittenSd;
    statusDoc["sd_combined_feature_rows_written"] = (uint32_t)combinedFeatureRowsWrittenSd;
    statusDoc["sd_combined_rows_failed"] = (uint32_t)combinedRowsFailedSd;
    statusDoc["cloud_last_error"] = cloudLastError;
    statusDoc["cloud_feature_publish_ok"] = (uint32_t)cloudFeaturePublishOk;
    statusDoc["cloud_feature_publish_failed"] = (uint32_t)cloudFeaturePublishFailed;
    statusDoc["cloud_publish_slow_count"] = (uint32_t)cloudPublishSlowCount;
    statusDoc["cloud_feature_stale_recoveries"] = (uint32_t)cloudFeatureStaleRecoveries;
    statusDoc["cloud_feature_stale_restarts"] = (uint32_t)cloudFeatureStaleRestarts;
    statusDoc["cloud_feature_age_ms"] = (uint32_t)(millis() - cloudLastFeaturePublishOkMs);
    statusDoc["cloud_reconnect_attempts"] = (uint32_t)cloudReconnectAttempts;
    statusDoc["cloud_network_resets"] = (uint32_t)cloudNetworkResets;
    statusDoc["cloud_last_progress_ms"] = (uint32_t)cloudLastProgressMs;
    statusDoc["pi_metrics_ok"] = (uint32_t)piMetricsOk;
    statusDoc["pi_metrics_fail"] = (uint32_t)piMetricsFail;
    statusDoc["pi_metrics_consecutive_fail"] = (uint32_t)piMetricsConsecutiveFail;
    statusDoc["pi_metrics_stale_recoveries"] = (uint32_t)piMetricsStaleRecoveries;
    uint32_t metricsAgeForStatus = latestPiMetricsAgeMs(millis());
    if (metricsAgeForStatus == UINT32_MAX) {
        statusDoc["pi_metrics_age_ms"] = nullptr;
    } else {
        statusDoc["pi_metrics_age_ms"] = metricsAgeForStatus;
    }
    statusDoc["pi_metrics_valid"] = latestPiMetricsValid ? true : false;

    // Rate scheduler status (visibility for ops)
    statusDoc["schedule_day"] = currentScheduleDay + 1;
    statusDoc["sample_period_us"] = currentSamplePeriodUs;
    statusDoc["sample_rate_hz"] = sampleRateHzForPeriodUs(currentSamplePeriodUs);
    if (experimentStartLoaded) {
        statusDoc["experiment_start_local"] = (uint32_t)experimentStartEpoch;
    }

    if (!isnan(lastTempC)) {
        statusDoc["temp_c"] = roundf(lastTempC * 10.0f) / 10.0f;
    }

    size_t len = serializeJson(statusDoc, statusPayload, sizeof(statusPayload));

    if (!mqtt.connected()) {
        noteCloudFailure("status_no_mqtt");
        return;
    }

    // retained = false; if true, AWS policy needs iot:RetainPublish
    bool ok = mqtt.publish(
        MQTT_TOPIC_STATUS,
        (const uint8_t*)statusPayload,
        len,
        false
    );

    Serial.printf("[MQTT] Status publish: %s | ok=%d | len=%u | sd=%d cooldown=%u raw=%u/%u feature=%u/%u err=%s\n",
                  connStatus,
                  ok ? 1 : 0,
                  (unsigned)len,
                  sdReady ? 1 : 0,
                  (unsigned)sdCooldownRemainingMs(),
                  (unsigned)rawChunksWrittenSd,
                  (unsigned)rawChunksFailedSd,
                  (unsigned)featureRowsWrittenSd,
                  (unsigned)featureRowsFailedSd,
                  sdLastError);

    if (ok) {
        noteCloudProgress("status_published");
    } else {
        noteCloudFailure("status_publish_failed");
    }
}

bool connectMQTT(uint32_t timeoutMs) {
    if (mqtt.connected()) return true;

    Serial.printf("[HEAP] Before TLS setup: free=%u, largest_block=%u\n",
                  ESP.getFreeHeap(),
                  ESP.getMaxAllocHeap());

    mqtt.setServer(AWS_ENDPOINT, AWS_PORT);
    mqtt.setBufferSize(MQTT_BUFFER_SIZE_BYTES);
    mqtt.setKeepAlive(60);
    mqtt.setSocketTimeout(CLOUD_MQTT_SOCKET_TIMEOUT_S);

    net.setCACert(AWS_CERT_CA);
    net.setCertificate(AWS_CERT_CRT);
    net.setPrivateKey(AWS_CERT_PRIVATE);
    net.setTimeout(CLOUD_NET_TIMEOUT_MS);

    Serial.printf("[MQTT] Buffer size set to %u bytes\n", MQTT_BUFFER_SIZE_BYTES);

    Serial.printf("[HEAP] After TLS setup:  free=%u, largest_block=%u\n",
                  ESP.getFreeHeap(),
                  ESP.getMaxAllocHeap());

    String lwtPayload =
        "{\"device_id\":\"" DEVICE_ID
        "\",\"serial_no\":\"" SERIAL_NO
        "\",\"status\":\"offline\"}";

    const unsigned long startMs = millis();
    while (!mqtt.connected() &&
           (uint32_t)(millis() - startMs) < timeoutMs) {
        esp_task_wdt_reset();

        sysState = CONNECTING;

        Serial.println("[MQTT] Connecting to AWS IoT...");
        Serial.printf("[MQTT] clientId=%s\n", DEVICE_ID);
        Serial.printf("[MQTT] endpoint=%s:%d\n", AWS_ENDPOINT, AWS_PORT);
        Serial.printf("[MQTT] LWT topic=%s\n", MQTT_TOPIC_LWT);

        unsigned long t0 = millis();

        // willRetain = false; if true, AWS policy needs iot:RetainPublish
        bool ok = mqtt.connect(
            DEVICE_ID,
            MQTT_TOPIC_LWT,
            1,
            false,
            lwtPayload.c_str()
        );

        unsigned long elapsed = millis() - t0;

        esp_task_wdt_reset();

        Serial.printf("[MQTT] connect() returned %d after %lu ms, state=%d\n",
                      ok ? 1 : 0,
                      elapsed,
                      mqtt.state());

        if (ok) {
            Serial.println("[MQTT] Connected");
            sysState = READY;
            noteCloudProgress("mqtt_connected");
            publishStatus("online");
            return true;
        }

        sysState = ERROR_STATE;

        Serial.printf("[MQTT] Failed, state=%d, retry in 3s\n", mqtt.state());
        char tlsError[128];
        const int tlsErrorCode = net.lastError(tlsError, sizeof(tlsError));
        Serial.printf("[TLS] lastError=%d %s\n", tlsErrorCode, tlsError);

        net.stop();

        for (int i = 0; i < 30; i++) {
            if ((uint32_t)(millis() - startMs) >= timeoutMs) break;
            esp_task_wdt_reset();
            updateLEDs();
            vTaskDelay(pdMS_TO_TICKS(100));
        }
    }

    Serial.printf("[MQTT] Connect timeout after %lu ms\n", (unsigned long)timeoutMs);
    noteCloudFailure("mqtt_timeout");
    net.stop();
    return false;
}

bool publishRawChunk(const RawChunkDesc& desc) {
#if ENABLE_RAW_BINARY_CHUNK && ENABLE_RAW_UPLOAD
    if (!rawChunkDescIsValid(desc)) {
        rawChunksFailedPublish++;
        return false;
    }

    if (!mqtt.connected()) {
        return false;
    }

    fillRawChunkHeader(desc);

    uint8_t* chunk = rawBuffers[desc.bufferIndex];
    const size_t totalBytes = sizeof(RawChunkHeader) + desc.payloadBytes;

    bool ok = mqtt.publish(
        MQTT_TOPIC_RAW,
        (const uint8_t*)chunk,
        totalBytes,
        false
    );

    if (ok) {
        rawChunksPublished++;

        Serial.printf("[MQTT] Published RAW chunk: seq=%u samples=%u bytes=%u topic=%s\n",
                      desc.seq,
                      desc.sampleCount,
                      (unsigned)totalBytes,
                      MQTT_TOPIC_RAW);
    } else {
        rawChunksFailedPublish++;

        Serial.printf("[MQTT] RAW publish failed: seq=%u state=%d bytes=%u\n",
                      desc.seq,
                      mqtt.state(),
                      (unsigned)totalBytes);
    }

    return ok;
#else
    (void)desc;
    return true;
#endif
}


bool publishPiMetrics() {
#if ENABLE_PI_METRICS_PACKET
    if (!mqtt.connected()) {
        noteCloudFailure("feature_no_mqtt");
        return false;
    }

    PiPowerMetrics m;
    uint32_t receivedMs = 0;

    if (!copyLatestPiMetrics(m, receivedMs)) {
        return false;
    }

    metricsDoc.clear();

    metricsDoc["schema_ver"] = 1;
    metricsDoc["device_id"] = DEVICE_ID;
    metricsDoc["serial_no"] = SERIAL_NO;
    metricsDoc["site_id"] = SITE_ID;
    metricsDoc["fw_version"] = FW_VERSION;
    metricsDoc["type"] = "pi_derived_metrics_1s";
    const uint64_t metricsUtcEpochMs = getUtcEpochMs();
    if (metricsUtcEpochMs != 0) {
        metricsDoc["utc_epoch_ms"] = metricsUtcEpochMs;
    }
    metricsDoc["received_ms"] = receivedMs;
    metricsDoc["pi_timestamp_us"] = m.timestamp_us;
    metricsDoc["n_samples"] = m.n_samples;
    metricsDoc["sample_rate_hz"] = m.sample_rate_hz;
    metricsDoc["frequency"] = roundf(m.frequency * 1000.0f) / 1000.0f;

    // ESP local temperature (DS18B20 on configured TEMP_SENSOR_PIN).
    if (isnan(lastTempC)) {
        metricsDoc["esp_temp_c"] = nullptr;
    } else {
        metricsDoc["esp_temp_c"] = roundf(lastTempC * 10.0f) / 10.0f;
    }

    // Pi enclosure temperature (DS18B20 inside Pi box).
    // -127.0 = sensor not detected → emit null so downstream knows.
    if (m.box_temp_c <= -100.0f) {
        metricsDoc["pi_box_temp_c"] = nullptr;
    } else {
        metricsDoc["pi_box_temp_c"] = roundf(m.box_temp_c * 10.0f) / 10.0f;
    }

    metricsDoc["free_heap"] = ESP.getFreeHeap();
    metricsDoc["uptime_s"] = (uint32_t)(millis() / 1000UL);

    JsonArray phases = metricsDoc.createNestedArray("phases");

    for (int i = 0; i < 3; i++) {
        JsonObject ph = phases.createNestedObject();
        ph["phase"] = i + 1;
        ph["Vrms"] = roundf(m.Vrms[i] * 1000.0f) / 1000.0f;
        ph["Irms"] = roundf(m.Irms[i] * 1000.0f) / 1000.0f;
        ph["P"] = roundf(m.P[i] * 1000.0f) / 1000.0f;
        ph["S"] = roundf(m.S[i] * 1000.0f) / 1000.0f;
        ph["PF"] = roundf(m.PF[i] * 1000.0f) / 1000.0f;
        ph["amp_V"] = roundf(m.amp_V[i] * 1000.0f) / 1000.0f;
        ph["amp_I"] = roundf(m.amp_I[i] * 1000.0f) / 1000.0f;
        ph["angle_V_rad"] = roundf(m.angle_V[i] * 1000.0f) / 1000.0f;
        ph["angle_I_rad"] = roundf(m.angle_I[i] * 1000.0f) / 1000.0f;
        ph["phase_diff_rad"] = roundf(m.phase_diff[i] * 1000.0f) / 1000.0f;
        ph["thd_V"] = roundf(m.thd_V[i] * 1000.0f) / 1000.0f;
        ph["thd_I"] = roundf(m.thd_I[i] * 1000.0f) / 1000.0f;
    }

    size_t len = serializeJson(metricsDoc, metricsPayload, sizeof(metricsPayload));

    if (len == 0 || len >= sizeof(metricsPayload)) {
        Serial.println("[MQTT] Metrics JSON buffer too small");
        return false;
    }

    unsigned long publishStartMs = millis();
    bool ok = mqtt.publish(
        MQTT_TOPIC_FEATURES,
        (const uint8_t*)metricsPayload,
        len,
        false
    );
    unsigned long publishElapsedMs = millis() - publishStartMs;

    if (publishElapsedMs > CLOUD_PUBLISH_SLOW_MS) {
        cloudPublishSlowCount++;
        Serial.printf("[MQTT] Feature publish slow: %lu ms\n", publishElapsedMs);
        resetCloudConnection("feature_publish_slow", false);
    }

    if (ok) {
        cloudFeaturePublishOk++;
        cloudLastFeaturePublishOkMs = millis();
        noteCloudProgress("feature_published");
        pulseDataLEDNonBlocking();

        Serial.printf("[MQTT] Published Pi metrics: freq=%.3fHz n=%u len=%u topic=%s\n",
                      m.frequency,
                      m.n_samples,
                      (unsigned)len,
                      MQTT_TOPIC_FEATURES);
    } else {
        cloudFeaturePublishFailed++;
        noteCloudFailure("feature_publish_failed");
        Serial.printf("[MQTT] Publish Pi metrics failed, state=%d\n", mqtt.state());
    }

    return ok;
#else
    return true;
#endif
}

bool publishAggregate(const AggregateSnapshot& snap) {
#if ENABLE_AGGREGATE_PUBLISH
    if (!mqtt.connected()) {
        noteCloudFailure("telemetry_no_mqtt");
        return false;
    }

    telemetryDoc.clear();

    telemetryDoc["schema_ver"] = 1;
    telemetryDoc["device_id"] = DEVICE_ID;
    telemetryDoc["serial_no"] = SERIAL_NO;
    telemetryDoc["site_id"] = SITE_ID;
    telemetryDoc["fw_version"] = FW_VERSION;
    telemetryDoc["type"] = "ef_aggregate_10s";
    const uint64_t telemetryUtcEpochMs = getUtcEpochMs();
    if (telemetryUtcEpochMs != 0) {
        telemetryDoc["utc_epoch_ms"] = telemetryUtcEpochMs;
    }
    telemetryDoc["sample_rate_hz"] = sampleRateHzForPeriodUs(currentSamplePeriodUs);
    telemetryDoc["window_ms"] = TELEMETRY_PUBLISH_INTERVAL_MS;
    telemetryDoc["samples"] = snap.samples;
    telemetryDoc["ok_reads"] = snap.okReads;
    telemetryDoc["failed_reads"] = snap.failedReads;
    telemetryDoc["rssi"] = WiFi.RSSI();
    telemetryDoc["free_heap"] = ESP.getFreeHeap();
    telemetryDoc["uptime_s"] = (uint32_t)(millis() / 1000UL);
    telemetryDoc["raw_chunks_published"] = (uint32_t)rawChunksPublished;
    telemetryDoc["raw_chunks_failed"] = (uint32_t)rawChunksFailedPublish;
    telemetryDoc["raw_chunks_dropped"] = (uint32_t)rawChunksDropped;

    if (!isnan(lastTempC)) {
        telemetryDoc["temp_c"] = roundf(lastTempC * 10.0f) / 10.0f;
    }

    // Channel order according to Ajarn's requester example:
    // [0] voltage Line 1
    // [1] current Line 1
    // [2] voltage Line 2
    // [3] current Line 2
    // [4] voltage Line 3
    // [5] current Line 3
    const char* names[N_CHANNELS] = {
        "v1", "i1", "v2", "i2", "v3", "i3"
    };

    JsonArray channels = telemetryDoc.createNestedArray("channels");

    for (int ch = 0; ch < N_CHANNELS; ch++) {
        JsonObject c = channels.createNestedObject();
        c["name"] = names[ch];

        if (snap.samples > 0) {
            const double n = (double)snap.samples;
            const double mean = snap.sum[ch] / n;
            const double rms = sqrt(snap.sumSq[ch] / n);

            c["mean"] = round(mean * 1000.0) / 1000.0;
            c["rms"] = round(rms * 1000.0) / 1000.0;
            c["min"] = round((double)snap.minVal[ch] * 1000.0) / 1000.0;
            c["max"] = round((double)snap.maxVal[ch] * 1000.0) / 1000.0;
        } else {
            c["mean"] = nullptr;
            c["rms"] = nullptr;
            c["min"] = nullptr;
            c["max"] = nullptr;
        }
    }

    size_t len = serializeJson(telemetryDoc, telemetryPayload, sizeof(telemetryPayload));

    if (len == 0 || len >= sizeof(telemetryPayload)) {
        Serial.println("[MQTT] Aggregate JSON buffer too small");
        return false;
    }

    bool ok = mqtt.publish(
        MQTT_TOPIC_TELEMETRY,
        (const uint8_t*)telemetryPayload,
        len,
        false
    );

    if (ok) {
        noteCloudProgress("telemetry_published");
        Serial.printf("[MQTT] Published aggregate: samples=%u ok=%u fail=%u len=%u\n",
                      snap.samples,
                      snap.okReads,
                      snap.failedReads,
                      (unsigned)len);
    } else {
        noteCloudFailure("telemetry_publish_failed");
        Serial.printf("[MQTT] Publish aggregate failed, state=%d\n", mqtt.state());
    }

    return ok;
#else
    (void)snap;
    return true;
#endif
}
#endif  // ENABLE_AWS_IOT

// ── Raw queues ──────────────────────────────────────────────────────────────
bool setupRawQueues() {
    rawFreeQueue = xQueueCreate(RAW_BUFFER_COUNT, sizeof(uint8_t));
    rawFilledQueue = xQueueCreate(RAW_BUFFER_COUNT, sizeof(RawChunkDesc));

    if (rawFreeQueue == nullptr || rawFilledQueue == nullptr) {
        Serial.println("[RAW] Queue create FAILED");
        return false;
    }

    for (uint8_t i = 0; i < RAW_BUFFER_COUNT; i++) {
        xQueueSend(rawFreeQueue, &i, 0);
    }

    Serial.printf("[RAW] Buffers ready: count=%u chunk_total=%u payload=%u header=%u\n",
                  RAW_BUFFER_COUNT,
                  (unsigned)RAW_CHUNK_TOTAL_BYTES,
                  (unsigned)RAW_PAYLOAD_BYTES,
                  (unsigned)sizeof(RawChunkHeader));

    return true;
}

// ── FreeRTOS tasks ──────────────────────────────────────────────────────────
void adcTask(void* parameter) {
    (void)parameter;

    Serial.printf("[ADC] Task started on core %d | initial target=%u Hz | RAW chunk=%u samples\n",
                  xPortGetCoreID(),
                  sampleRateHzForPeriodUs(currentSamplePeriodUs),
                  RAW_CHUNK_SAMPLES);

    esp_task_wdt_add(NULL);

    int16_t rawValues[N_CHANNELS];

    uint8_t activeBufferIndex = 0;
    bool hasBuffer = false;
    uint16_t samplePos = 0;
    uint32_t seq = 0;
    uint64_t chunkStartMs = 0;
    uint64_t chunkStartUtcMs = 0;
    int64_t chunkFirstSampleUs = 0;
    uint16_t chunkSampleRateHz = sampleRateHzForPeriodUs(currentSamplePeriodUs);

    uint32_t wdtFeedCounter = 0;
    uint8_t schedulingYieldCounter = 0;
    uint32_t consecutiveFails = 0;
    uint32_t lastFailLogMs = 0;
    uint32_t lastMetricsRequestMs = 0;

    int64_t nextSampleUs = esp_timer_get_time();

    for (;;) {
        // ── Check for pending rate change from cloudTask ──
        // Drains 1 entry per loop iteration; if multiple queued, latest wins
        // on subsequent iterations. 7-byte UART write is ~28 µs at 2 Mbaud
        // — negligible compared to sample period.
        {
            uint32_t pendingPeriod = 0;
            if (rateChangeQueue != nullptr &&
                xQueueReceive(rateChangeQueue, &pendingPeriod, 0) == pdTRUE) {

                uint8_t pkt[7];
                buildRateSetPacket(pendingPeriod, pkt);

                // Make sure any pending Pi response is cleared, then send
                while (piSerial.available()) piSerial.read();
                piSerial.write(pkt, 7);
                piSerial.flush();

                Serial.printf("[RATE] Sent set-rate packet to Pi: period=%lu us\n",
                              (unsigned long)pendingPeriod);

                // Pi sends 3-byte ACK [0xFB][period_LSB][0xFD] — consume if present
                // (optional, just to keep UART clean)
                uint32_t ackWait = micros();
                while ((uint32_t)(micros() - ackWait) < 5000UL) {
                    if (piSerial.available() >= 3) {
                        uint8_t ack[3];
                        piSerial.readBytes(ack, 3);
                        if (ack[0] == 0xFB && ack[2] == 0xFD) {
                            // Got ACK — fine
                        }
                        break;
                    }
                    delayMicroseconds(50);
                }

                currentSamplePeriodUs = pendingPeriod;

                if (hasBuffer && samplePos > 0) {
                    Serial.printf("[RATE] Dropping partial RAW chunk before rate switch: samples=%u\n",
                                  samplePos);
                    xQueueSend(rawFreeQueue, &activeBufferIndex, 0);
                    hasBuffer = false;
                    samplePos = 0;
                }

                // Reset sample timing so we don't try to "catch up" stale samples
                nextSampleUs = esp_timer_get_time();
            }
        }

        if (!hasBuffer) {
            if (xQueueReceive(rawFreeQueue, &activeBufferIndex, portMAX_DELAY) == pdTRUE) {
                hasBuffer = true;
                samplePos = 0;
                chunkStartMs = millis();
                chunkStartUtcMs = getUtcEpochMs();
                chunkSampleRateHz = sampleRateHzForPeriodUs(currentSamplePeriodUs);

                // Clear current buffer
                memset(rawBuffers[activeBufferIndex], 0, RAW_CHUNK_TOTAL_BYTES);
            }
        }

        unsigned long nowMetricsMs = millis();
#if ENABLE_PI_METRICS_PACKET
        const bool metricsDue =
            (uint32_t)(nowMetricsMs - lastMetricsRequestMs) >= METRICS_REQUEST_INTERVAL_MS;
        PiPowerMetrics metricsFromPi;
        bool metricsReceived = false;

        const bool ok = metricsDue
            ? requestPiSampleRawAndMetrics(rawValues, &metricsFromPi, &metricsReceived)
            : requestPiSampleRaw(rawValues);
#else
        const bool ok = requestPiSampleRaw(rawValues);
#endif

        if (ok) {
            rawSamplesOk++;
            consecutiveFails = 0;

            // Timestamp the first successful frame, not buffer allocation.
            // Queue availability must never distort the raw time axis.
            const int64_t capturedUs = esp_timer_get_time();
            if (samplePos == 0) {
                chunkStartMs = millis();
                chunkStartUtcMs = getUtcEpochMs();
                chunkFirstSampleUs = capturedUs;
                chunkSampleRateHz = sampleRateHzForPeriodUs(currentSamplePeriodUs);
            }

            // Store raw int16 into active chunk payload
            if (hasBuffer) {
                uint8_t* payload = rawBuffers[activeBufferIndex] + sizeof(RawChunkHeader);
                const size_t offset =
                    (size_t)samplePos * (size_t)N_CHANNELS * sizeof(int16_t);

                memcpy(payload + offset, rawValues, N_CHANNELS * sizeof(int16_t));
                samplePos++;
            }

            // Also update aggregate statistics
            addSampleToAggregateRaw(rawValues);

            if (hasBuffer && samplePos >= RAW_CHUNK_SAMPLES) {
                RawChunkDesc desc;
                desc.bufferIndex = activeBufferIndex;
                desc.seq = seq++;
                desc.timestampMs = chunkStartMs;
                desc.utcEpochMs = chunkStartUtcMs;
                desc.sampleCount = RAW_CHUNK_SAMPLES;
                desc.sampleRateHz = chunkSampleRateHz;
                if (chunkFirstSampleUs > 0 && capturedUs > chunkFirstSampleUs) {
                    const uint64_t elapsedUs = (uint64_t)(capturedUs - chunkFirstSampleUs);
                    const uint64_t numerator =
                        (uint64_t)(RAW_CHUNK_SAMPLES - 1U) * 1000000ULL;
                    const uint64_t measuredHz = (numerator + (elapsedUs / 2ULL)) / elapsedUs;
                    if (measuredHz > 0 && measuredHz <= 65535ULL) {
                        desc.sampleRateHz = (uint16_t)measuredHz;
                    }
                }
                desc.payloadBytes = RAW_PAYLOAD_BYTES;

                if (xQueueSend(rawFilledQueue, &desc, 0) != pdTRUE) {
                    rawChunksDropped++;
                    Serial.printf("[RAW] Filled queue full, drop chunk seq=%u\n", desc.seq);

                    // Return buffer to free queue so system can continue
                    xQueueSend(rawFreeQueue, &activeBufferIndex, 0);
                }

                hasBuffer = false;
                samplePos = 0;
            }
        } else {
            rawSamplesFailed++;
            addFailedRead();
            consecutiveFails++;

            if (consecutiveFails >= 1000) {
                unsigned long nowMs = millis();

                if (nowMs - lastFailLogMs > 5000) {
                    Serial.printf("[ADC] Pi/EF not responding (consecutive fails=%u)\n",
                                  consecutiveFails);
                    lastFailLogMs = nowMs;
                }

                vTaskDelay(pdMS_TO_TICKS(10));
                esp_task_wdt_reset();

                nextSampleUs = esp_timer_get_time();
                continue;
            }
        }

        // Request derived metrics at the configured feature cadence by piggybacking
        // them onto the same raw UART transaction. This keeps raw at 1 kHz while
        // removing the second request/response pair that used to contend on UART.
#if ENABLE_PI_METRICS_PACKET
        if (metricsDue) {
            if (metricsReceived) {
                uint32_t metricsReceivedMs = millis();
                storeLatestPiMetrics(metricsFromPi);
                queueFeatureMetricsForSd(metricsFromPi, metricsReceivedMs);
                Serial.printf("[METRICS] Pi metrics received: fs=%u n=%u freq=%.3fHz box=%.1fC ok=%u fail=%u\n",
                              metricsFromPi.sample_rate_hz,
                              metricsFromPi.n_samples,
                              metricsFromPi.frequency,
                              metricsFromPi.box_temp_c,
                              (uint32_t)piMetricsOk,
                              (uint32_t)piMetricsFail);
            } else {
                piMetricsFail++;
                piMetricsConsecutiveFail++;
                if ((piMetricsConsecutiveFail % 5U) == 0U) {
                    while (piSerial.available()) {
                        piSerial.read();
                    }
                    Serial.printf("[METRICS] Request failing: consecutive=%u total_fail=%u; UART flushed\n",
                                  (uint32_t)piMetricsConsecutiveFail,
                                  (uint32_t)piMetricsFail);
                }
            }
            lastMetricsRequestMs = nowMetricsMs;
        }
#endif

        if (++wdtFeedCounter >= 100) {
            esp_task_wdt_reset();
            wdtFeedCounter = 0;
        }

        const uint32_t samplePeriodUs = currentSamplePeriodUs;
        nextSampleUs += samplePeriodUs;

        // adcTask owns Core 1. Yield periodically, not once per sample, so a
        // scheduler handoff cannot consume a material part of a 1 ms budget.
        if (++schedulingYieldCounter >= 32U) {
            taskYIELD();
            schedulingYieldCounter = 0;
        }

        int64_t waitUs = nextSampleUs - esp_timer_get_time();

        if (waitUs > 0) {
            delayMicroseconds((uint32_t)waitUs);
        } else if (waitUs < -(int64_t)samplePeriodUs) {
            // If late more than one period, resync to avoid accumulated lag.
            nextSampleUs = esp_timer_get_time();
        }
    }
}

void sdWriterTask(void* parameter) {
    (void)parameter;

    Serial.printf("[SD] Writer task started on core %d | raw=%d feature=%d combined=%d\n",
                  xPortGetCoreID(),
                  ENABLE_RAW_SD_LOG ? 1 : 0,
                  ENABLE_FEATURE_SD_LOG ? 1 : 0,
                  ENABLE_COMBINED_SD_LOG ? 1 : 0);

    esp_task_wdt_add(NULL);

    for (;;) {
        bool didWork = false;

#if ENABLE_SD_CARD_LOG
        if (!sdReady) {
            serviceSdCardReconnect();
        }
#endif

#if ENABLE_SD_CARD_LOG && (ENABLE_RAW_SD_LOG || ENABLE_COMBINED_SD_LOG)
        RawChunkDesc desc;
        if (xQueueReceive(rawFilledQueue, &desc, pdMS_TO_TICKS(20)) == pdTRUE) {
            bool rawOk = true;
            bool combinedOk = true;
#if ENABLE_RAW_SD_LOG
            rawOk = writeRawChunkToSd(desc);
#endif
#if ENABLE_COMBINED_SD_LOG
            combinedOk = writeCombinedSampleChunkToSd(desc);
#endif

            // SD is intentionally fail-safe. A card/socket problem must not
            // mark the whole device as failed or slow the AWS feature path.
            (void)rawOk;
            (void)combinedOk;

            xQueueSend(rawFreeQueue, &desc.bufferIndex, 0);
            didWork = true;
            esp_task_wdt_reset();
        }
#endif

#if ENABLE_SD_CARD_LOG && (ENABLE_FEATURE_SD_LOG || ENABLE_COMBINED_SD_LOG)
        FeatureSdRecord rec;
        while (featureSdQueue != nullptr && xQueueReceive(featureSdQueue, &rec, 0) == pdTRUE) {
            bool featureOk = true;
            bool combinedOk = true;
#if ENABLE_FEATURE_SD_LOG
            featureOk = writeFeatureRecordToSd(rec);
#endif
#if ENABLE_COMBINED_SD_LOG
            combinedOk = writeCombinedFeatureRecordToSd(rec);
#endif

            // Feature SD logging is local-only; AWS feature publish continues.
            (void)featureOk;
            (void)combinedOk;

            didWork = true;
            esp_task_wdt_reset();
        }
#endif

        if (!didWork) {
            vTaskDelay(pdMS_TO_TICKS(20));
        }

        esp_task_wdt_reset();
    }
}

// Local-recorder service. This is the only Core 0 companion task in the
// production SD-only build: it maintains time and LEDs, but never opens TLS,
// constructs an MQTT session, or consumes the raw-buffer queue.
void localServiceTask(void* parameter) {
    (void)parameter;

    Serial.printf("[LOCAL] Service task started on core %d | AWS/MQTT disabled\n",
                  xPortGetCoreID());

    esp_task_wdt_add(NULL);
    unsigned long lastReconnectAttemptMs = millis() - CLOUD_RECONNECT_INTERVAL_MS;

    for (;;) {
        const unsigned long now = millis();
        refreshLocalLinkState();

        if (!wifiLinkUp &&
            now - lastReconnectAttemptMs >= CLOUD_RECONNECT_INTERVAL_MS) {
            connectWiFi(WIFI_CONNECT_TIMEOUT_MS);
            lastReconnectAttemptMs = millis();
        }

        updateTemperature();
        updateLEDs();
        rateSchedulerTick();
        esp_task_wdt_reset();
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

#if ENABLE_AWS_IOT
void cloudTask(void* parameter) {
    (void)parameter;

    Serial.printf("[CLOUD] Task started on core %d\n", xPortGetCoreID());

    esp_task_wdt_add(NULL);
    esp_task_wdt_reset();

    cloudLastProgressMs = millis();
    cloudLastReconnectAttemptMs = millis() - CLOUD_RECONNECT_INTERVAL_MS;
    unsigned long lastTelemetryMs = millis();
    unsigned long lastStatusMs = millis();
    unsigned long lastFeaturePublishAttemptMs = 0;
    uint32_t lastPublishedMetricsReceivedMs = 0;
    cloudLastFeaturePublishOkMs = millis();
    cloudLastFeatureRecoveryMs = millis();

    for (;;) {
        esp_task_wdt_reset();
        unsigned long now = millis();

#if !ENABLE_AWS_IOT
        // SD-only mode: retain WiFi/NTP for correct daily filenames, but never
        // construct a TLS/MQTT session while the AWS side is intentionally off.
        if (WiFi.status() != WL_CONNECTED &&
            now - cloudLastReconnectAttemptMs >= CLOUD_RECONNECT_INTERVAL_MS) {
            connectWiFi(WIFI_CONNECT_TIMEOUT_MS);
            cloudLastReconnectAttemptMs = millis();
        }

        updateTemperature();
        updateLEDs();
        rateSchedulerTick();
        vTaskDelay(pdMS_TO_TICKS(20));
        continue;
#endif

        if (WiFi.status() != WL_CONNECTED) {
            sysState = ERROR_STATE;
            if (now - cloudLastReconnectAttemptMs >= CLOUD_RECONNECT_INTERVAL_MS) {
                connectWiFi(WIFI_CONNECT_TIMEOUT_MS);
                now = millis();
                cloudLastReconnectAttemptMs = now;
            }
            esp_task_wdt_reset();
        } else if (!mqtt.connected()) {
            sysState = ERROR_STATE;
            if (now - cloudLastReconnectAttemptMs >= CLOUD_RECONNECT_INTERVAL_MS) {
                const bool mqttOk = connectMQTT(MQTT_CONNECT_TIMEOUT_MS);
                now = millis();
                cloudLastReconnectAttemptMs = now;
                if (!mqttOk && WiFi.status() == WL_CONNECTED) {
                    resetCloudConnection("mqtt_connect_failed_reset_wifi", true);
                    now = millis();
                    cloudLastProgressMs = now;
                    cloudLastReconnectAttemptMs = now;
                }
            }
            esp_task_wdt_reset();
        }

        if (mqtt.connected()) {
            mqtt.loop();
        }

        now = millis();
        if (now - cloudLastProgressMs >= CLOUD_STALL_RESET_MS) {
            const bool resetWifi = (WiFi.status() != WL_CONNECTED);
            resetCloudConnection("cloud_stall_timeout", resetWifi);
            now = millis();
            cloudLastProgressMs = now;
            cloudLastReconnectAttemptMs = now;
        }

#if ENABLE_PI_METRICS_PACKET
        if (mqtt.connected() &&
            now - cloudLastFeaturePublishOkMs >= FEATURE_STALE_RECOVERY_MS &&
            now - cloudLastFeatureRecoveryMs >= FEATURE_STALE_RECOVERY_MS) {
            const uint32_t featureAgeMs = now - cloudLastFeaturePublishOkMs;
            const uint32_t metricsAgeMs = latestPiMetricsAgeMs(now);
            cloudFeatureStaleRecoveries++;
            cloudLastFeatureRecoveryMs = now;

            if (metricsAgeMs == UINT32_MAX || metricsAgeMs >= PI_METRICS_STALE_MS) {
                piMetricsStaleRecoveries++;
                invalidateLatestPiMetrics();
                clearPiSerialRx();
                Serial.printf("[RECOVERY] Feature stale: age=%u ms metrics_age=%u ms; metrics invalidated and UART flushed, cloud left connected\n",
                              featureAgeMs,
                              metricsAgeMs);
                cloudLastProgressMs = now;
            } else {
                Serial.printf("[RECOVERY] Feature stale: age=%u ms metrics_age=%u ms; resetting cloud\n",
                              featureAgeMs,
                              metricsAgeMs);
                resetCloudConnection("feature_stale_recovery", true);
                now = millis();
                cloudLastProgressMs = now;
                cloudLastReconnectAttemptMs = now;
            }

            if ((metricsAgeMs != UINT32_MAX && metricsAgeMs < PI_METRICS_STALE_MS) &&
                featureAgeMs >= FEATURE_STALE_RESTART_MS) {
                cloudFeatureStaleRestarts++;
                Serial.printf("[RECOVERY] Feature stale for %u ms; restarting ESP32\n", featureAgeMs);
                delay(100);
                ESP.restart();
            }
        }
#endif

        updateTemperature();
        updateLEDs();
        rateSchedulerTick();   // NEW: check 18:00 boundary, queue rate change if needed

        // Publish any completed raw chunk. When SD logging consumes raw samples,
        // sdWriterTask owns rawFilledQueue so cloud delays cannot back up sampling.
#if ENABLE_SD_CARD_LOG && (ENABLE_RAW_SD_LOG || ENABLE_COMBINED_SD_LOG)
        if (!sdWriterTaskStarted) {
            RawChunkDesc desc;
            while (xQueueReceive(rawFilledQueue, &desc, 0) == pdTRUE) {
                rawChunksDropped++;
                xQueueSend(rawFreeQueue, &desc.bufferIndex, 0);
            }
        }
#else
        RawChunkDesc desc;
        while (xQueueReceive(rawFilledQueue, &desc, 0) == pdTRUE) {
            bool ok = publishRawChunk(desc);

            // Return buffer to free queue after publish attempt
            xQueueSend(rawFreeQueue, &desc.bufferIndex, 0);

            if (!ok) {
                sysState = ERROR_STATE;
            } else {
                sysState = READY;
            }

            esp_task_wdt_reset();
            mqtt.loop();
        }
#endif

#if ENABLE_PI_METRICS_PACKET
        PiPowerMetrics m;
        uint32_t receivedMs = 0;
        if (copyLatestPiMetrics(m, receivedMs)) {
            uint32_t publishKey = (m.timestamp_us != 0) ? m.timestamp_us : receivedMs;
            if (mqtt.connected() &&
                publishKey != lastPublishedMetricsReceivedMs &&
                now - lastFeaturePublishAttemptMs >= FEATURE_PUBLISH_RETRY_MS) {
                lastFeaturePublishAttemptMs = now;
                if (!publishPiMetrics()) {
                    sysState = ERROR_STATE;
                } else {
                    sysState = READY;
                    lastPublishedMetricsReceivedMs = publishKey;
                }
            }
        }
#endif

        if (now - lastTelemetryMs >= TELEMETRY_PUBLISH_INTERVAL_MS) {
            if (mqtt.connected()) {
                AggregateSnapshot snap = takeAggregateSnapshotAndReset();

                if (!publishAggregate(snap)) {
                    sysState = ERROR_STATE;
                } else {
                    sysState = READY;
                }
            }

            lastTelemetryMs = now;
        }

        if (now - lastStatusMs >= STATUS_INTERVAL_MS) {
            if (mqtt.connected()) {
                publishStatus("online");
            }
            lastStatusMs = now;
        }

        vTaskDelay(pdMS_TO_TICKS(10));
    }
}
#endif  // ENABLE_AWS_IOT

// ── Setup / Loop ────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.printf("\n[BOOT] PowerLens %s | %s | FW %s\n",
                  DEVICE_ID,
                  SERIAL_NO,
                  FW_VERSION);

#if ENABLE_PI_METRICS_PACKET
    Serial.println("[BOOT] Mode: RAW 1kHz binary 250ms chunk + SD raw log + AWS feature packet");
    Serial.println("[BOOT] ESP32 requests Pi/EF raw+metrics over UART, stores raw locally, and publishes features to AWS IoT");
#else
    Serial.println("[BOOT] Mode: RAW-only 1kHz binary 250ms chunk + SD raw log; Pi feature compute disabled");
    Serial.println("[BOOT] ESP32 requests Pi/EF raw over UART and stores locally; feature analysis is offline from SD raw");
#endif

    printResetReason();

    setupWatchdog();

    Serial.println("[LED] Local ESP32 GPIO drive disabled; forwarding LED state to Pi only");

#if ENABLE_PI_ADC_TASK
    piSerial.setRxBufferSize(1024);
    piSerial.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);

    Serial.printf("[UART] Pi/EF request-response ready at %lu baud (RX=%d, TX=%d)\n",
                  (unsigned long)UART_BAUD,
                  UART_RX_PIN,
                  UART_TX_PIN);
#else
    Serial.println("[UART] Disabled");
#endif

    if (TEMP_SENSOR_PIN < 0) {
        tempSensorEnabled = false;
        Serial.println("[DS18B20] Disabled by config (TEMP_SENSOR_PIN < 0); using pi_box_temp_c from Pi instead");
    } else if (TEMP_SENSOR_PIN >= 34) {
        tempSensorEnabled = false;
        Serial.printf("[DS18B20] Disabled: GPIO%d is input-only on ESP32\n", TEMP_SENSOR_PIN);
    } else {
        pinMode(TEMP_SENSOR_PIN, INPUT_PULLUP);
        ds18b20.begin();
        ds18b20.setWaitForConversion(false);
        const uint8_t deviceCount = ds18b20.getDeviceCount();
        Serial.printf("[DS18B20] Enabled on GPIO%d | detected=%u\n", TEMP_SENSOR_PIN, deviceCount);
        if (deviceCount == 0) {
            Serial.println("[DS18B20] WARNING: no sensor detected on ESP bus; check DATA/VCC/GND and 4.7k pull-up");
        }
    }

    setupRawQueues();

    // NEW: create rate change queue (capacity 4 — should never fill in normal use)
    rateChangeQueue = xQueueCreate(4, sizeof(uint32_t));
    if (rateChangeQueue == nullptr) {
        Serial.println("[RATE] Queue create FAILED — rate scheduling disabled");
    } else {
        Serial.println("[RATE] Rate change queue ready");
    }

    Serial.printf("[BOOT] Free heap before tasks: %u\n", ESP.getFreeHeap());

#if ENABLE_SD_CARD_LOG
    setupSdCard();

    BaseType_t sdResult = xTaskCreatePinnedToCore(
        sdWriterTask,
        "sdWriterTask",
        8192,
        nullptr,
        1,
        nullptr,
        0
    );

    sdWriterTaskStarted = (sdResult == pdPASS);

    Serial.printf("[BOOT] sdWriterTask create: %s (heap=%u)\n",
                  sdWriterTaskStarted ? "OK" : "FAILED",
                  ESP.getFreeHeap());
#endif

#if ENABLE_AWS_IOT
    BaseType_t serviceResult = xTaskCreatePinnedToCore(
        cloudTask,
        "cloudTask_AWS",
        16384,
        nullptr,
        3,
        nullptr,
        0
    );
    Serial.printf("[BOOT] cloudTask create: %s (heap=%u)\n",
                  serviceResult == pdPASS ? "OK" : "FAILED",
                  ESP.getFreeHeap());
#else
    BaseType_t serviceResult = xTaskCreatePinnedToCore(
        localServiceTask,
        "localService",
        8192,
        nullptr,
        1,
        nullptr,
        0
    );
    Serial.printf("[BOOT] localService create: %s (heap=%u)\n",
                  serviceResult == pdPASS ? "OK" : "FAILED",
                  ESP.getFreeHeap());
#endif

    delay(200);

#if ENABLE_PI_ADC_TASK
    BaseType_t adcResult = xTaskCreatePinnedToCore(
        adcTask,
        "adcTask_RAW_1kHz",
        5120,
        nullptr,
        3,
        nullptr,
        1
    );

    Serial.printf("[BOOT] adcTask create: %s (heap=%u)\n",
                  adcResult == pdPASS ? "OK" : "FAILED",
                  ESP.getFreeHeap());
#else
    Serial.println("[ADC] Disabled");
#endif

    Serial.println("[BOOT] Tasks created");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
