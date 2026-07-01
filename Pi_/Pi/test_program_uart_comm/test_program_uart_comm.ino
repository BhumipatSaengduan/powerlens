#define usbdebug 0

#define endofline 0x0D
#define UART_BAUD_RATE 2000000
const int LED1_pin = 13;  // arranged left to right
const int LED2_pin = 14;  // in the orientation where
const int LED3_pin = 15;  // pi pico is on the right hand side, viewed from top

#define transmit_data_channels 6
// variables that will be shared across two cores
static volatile bool tc_dataready_flag = false
static volatile bool tc_busy_flag = true;
static volatile bool tc_has_sample = false;
// Even values are stable snapshots. Odd values mean Core 0 is publishing.
// This prevents Core 1 from sending a torn six-channel frame to the ESP32.
static volatile uint32_t tc_sample_generation = 0;
static int16_t tc_voltdata_int16t[transmit_data_channels];
static int16_t voltdata_int16t_core0txbuffer[transmit_data_channels];
static int16_t voltdata_int16t_core1rxbuffer[transmit_data_channels];
static int16_t voltdata_int16t_printbuffer[transmit_data_channels];
volatile unsigned int count = 0;  // not used in production
static volatile double mult;
// the unit of tc_voltdata_int16t is (x1/5k)(volts)

#include "MCP3x6x_mod.h"
#include "SPI.h"
#include "power_metrics.h"
#include <math.h>
#include <string.h>

extern volatile bool adc_ready;

#ifndef FEATURE_SERIAL_PRINT
#define FEATURE_SERIAL_PRINT 0
#endif

#ifndef ENABLE_PI_FEATURE_COMPUTE
#define ENABLE_PI_FEATURE_COMPUTE 0
#endif

#ifndef FEATURE_PRINT_INTERVAL_MS
#define FEATURE_PRINT_INTERVAL_MS 1000UL
#endif

#ifndef FEATURE_RAW_WINDOW_DEBUG
#define FEATURE_RAW_WINDOW_DEBUG 0
#endif

#ifndef RAW_SERIAL_PRINT
#define RAW_SERIAL_PRINT 0
#endif

#ifndef WAIT_FOR_USB_SERIAL
#define WAIT_FOR_USB_SERIAL 0
#endif

#ifndef FEATURE_WINDOW_SAMPLES
#define FEATURE_WINDOW_SAMPLES 1024
#endif

#ifndef FEATURE_RING_SIZE
#define FEATURE_RING_SIZE 2048
#endif

#ifndef PI_BOX_TEMP_UNAVAILABLE_C
#define PI_BOX_TEMP_UNAVAILABLE_C (-127.0f)
#endif

static const uint8_t EF_RAW_START_BYTE = 0xA5;
static const uint8_t EF_RAW_TYPE_SAMPLE = 0x01;
static const uint8_t EF_RAW_END_BYTE = 0x5A;
static const uint8_t EF_RAW_PAYLOAD_BYTES = transmit_data_channels * 2;
static const uint8_t EF_RAW_FRAME_BYTES = 1 + 1 + 2 + EF_RAW_PAYLOAD_BYTES + 2 + 1;

static const uint8_t METRICS_START_BYTE = 0xAA;
static const uint8_t METRICS_TYPE_DERIVED = 0x02;
static const uint8_t METRICS_END_BYTE = 0x55;
static const uint16_t METRICS_PAYLOAD_BYTES = 160;
static const uint16_t METRICS_PACKET_BYTES = 1 + 1 + 2 + METRICS_PAYLOAD_BYTES + 2 + 1;

static const uint8_t RATE_SET_PACKET_START = 0xFC;
static const uint8_t RATE_SET_PACKET_END = 0xFD;
static const uint8_t RATE_ACK_START = 0xFB;
static const uint8_t RATE_ACK_BYTES = 3;
static const uint8_t RATE_PACKET_REMAINING_BYTES = 5;
static const uint8_t COMMAND_PACKET_BYTES = 2;
static const uint32_t UART_COMMAND_TIMEOUT_MS = 2;
static const uint32_t RAW_DATAREADY_WAIT_TIMEOUT_US = 10000UL;

static_assert((FEATURE_RING_SIZE & (FEATURE_RING_SIZE - 1)) == 0,
              "FEATURE_RING_SIZE must be a power of two");

static volatile int16_t feature_ring_buffer[FEATURE_RING_SIZE][transmit_data_channels];
static volatile uint32_t feature_ring_head = 0;
static int16_t feature_snapshot[FEATURE_WINDOW_SAMPLES * transmit_data_channels];
static int16_t feature_snapshot_filter[FEATURE_WINDOW_SAMPLES * transmit_data_channels];
static PowerMetrics latest_feature_metrics;
static bool latest_feature_valid = false;
static uint32_t feature_fs_last_head = 0;
static uint32_t feature_fs_last_us = 0;
static float feature_estimated_fs = 1000.0f;

#if defined ARDUINO_AVR_PROMICRO8
MCP3561 mcp(10);
#elif defined ARDUINO_GRAND_CENTRAL_M4
SPIClass mySPI = SPIClass(&sercom5, 125, 126, 99, SPI_PAD_0_SCK_3, SERCOM_RX_PAD_2);
MCP3561 mcp(98, &mySPI);
#elif defined ADAFRUIT_METRO_M0_EXPRESS
SPIClass mySPI(&sercom1, 12, 13, 11, SPI_PAD_0_SCK_1, SERCOM_RX_PAD_3);
MCP3561 mcp(10, &mySPI, 11, 12, 13);

// #elif
// todo: might need further cases, didn't check for all boards
#else
// MCP3561 mcp;

#define pico2w 1
#define esp32 0
#if (pico2w)
const uint8_t pin_CS = 5;
const uint8_t pin_MOSI = 3;
const uint8_t pin_MISO = 4;
const uint8_t pin_CLK = 6;
const uint8_t pin_SPI_reset = 8;
const uint8_t pin_DC = 9;
const uint8_t pin_IRQ = 2;
const uint8_t pin_MCLK = 1;

MCP3564 mcp(
    pin_IRQ,
    pin_MCLK,
    MCP3564_DEVICE_TYPE,
    pin_CS,
    &SPI,
    pin_MOSI,
    pin_MISO,
    pin_CLK
);

#elif (esp32)
const uint8_t pin_CS = 15;
const uint8_t pin_MOSI = 13;
const uint8_t pin_MISO = 12;
const uint8_t pin_CLK = 14;
#define SD_CS_PIN 15
#define SD_CLK_PIN 14
#define SD_MOSI_PIN 13
#define SD_MISO_PIN 12
SPIClass SPI2(HSPI);
const uint8_t pin_SPI_reset = 8;
const uint8_t pin_DC = 9;
const uint8_t pin_IRQ = 2;
const uint8_t pin_MCLK = 1;

MCP3564 mcp(pin_IRQ, pin_MCLK, pin_CS, &SPI2, pin_MOSI, pin_MISO, pin_CLK);
#endif

const uint8_t pin_BJT1 = 13;
const uint8_t pin_BJT2 = 14;
const uint8_t pin_BJT3 = 15;

const uint8_t pin_SCL = 21;
const uint8_t pin_SDA = 20;

const int read_channels = 7;
static int32_t raw_read_bits[read_channels];
// MCP3x6x::mux_t MCP_CH_arr[8] = { MCP_CH0, MCP_CH1, MCP_CH2, MCP_CH3, MCP_CH4, MCP_CH5, MCP_CH6 , MCP_CH7};

// first one reads last channel for some reason @_@ skip it when collecting data
// read sequence: [dont care] V1 C1 V2 C2 V3 C3 [dont care]
// MCP3x6x::mux_t MCP_CH_arr[8] = { MCP_CH0, MCP_CH1, MCP_CH4, MCP_CH2, MCP_CH5, MCP_CH3, MCP_CH6, MCP_CH7 };
MCP3x6x::mux_t MCP_CH_arr[] = { MCP_CH0, MCP_CH1, MCP_CH2, MCP_CH3, MCP_CH4, MCP_CH5, MCP_CH6, MCP_CH7 };
//const uint8_t MCP_CH_arr[read_channels] = {MCP_CH1, MCP_CH0, MCP_CH3, MCP_CH2, MCP_CH5, MCP_CH4};
// MCP3564(const uint8_t pinCS = SS,
//         SPIClass *theSPI = &SPI,
//         const uint8_t pinMOSI = MOSI,
//         const uint8_t pinMISO = MISO,
//         const uint8_t pinCLK = SCK)
//     : MCP3x6x(MCP3564_DEVICE_TYPE, pinCS, theSPI, pinMOSI, pinMISO, pinCLK){};

// MCP3564 mcp(pin_IRQ, pin_MCLK, pin_CS, &SPI, pin_MOSI, pin_MISO, pin_CLK);

//MCP3564 mcp(pin_CS);
//MCP3x6x mcp(MCP3564_DEVICE_TYPE, pin_CS, &SPI, pin_MOSI, pin_MISO, pin_CLK);

#endif

void mcp_wrapper() {
  mcp.IRQ_handler();
}

volatile bool IRQ_flag = false;

void setup() { 
#if (pico2w)
  SPI.setRX(pin_MISO);
  SPI.setTX(pin_MOSI);
  SPI.setSCK(pin_CLK);
  SPI.setCS(pin_CS);
#endif

  if (!mcp.begin(0, 3.30)) {
#if (usbdebug)
    Serial.println("failed to initialize MCP");
#endif
    while (1);
  }

  // เปิด scan mode ทั้ง 6 channel (skip CH0 เพราะอ่านค่าแปลก)
  mcp.enableScanChannel(MCP_CH1);
  mcp.enableScanChannel(MCP_CH2);
  mcp.enableScanChannel(MCP_CH3);
  mcp.enableScanChannel(MCP_CH4);
  mcp.enableScanChannel(MCP_CH5);
  mcp.enableScanChannel(MCP_CH6);
  mcp.startContinuous();

  delay(10);
}

unsigned long previousMillis = 0;
const long interval = 1000;
double voltage = 0.0;
unsigned long micro2convert = 0;
unsigned long micro2read = 0;
unsigned long lastmicro = 0;

int32_t readch(MCP3x6x::mux_t MCP_CH) {
  static int32_t buff;
  // read the input on default analog channel:
  IRQ_flag = false;
  lastmicro = micros();
  mcp.convertMux(MCP_CH);
  while (!IRQ_flag) {  // wait for ADC to finish converting
    ;
  }
  micro2convert = micros() - lastmicro;
  lastmicro = micros();
  buff = mcp.analogReadMux(MCP_CH);
  // while (digitalRead(pin_IRQ)) { // wait for IRQ pin to return to high state
  //   ;
  // }
  micro2read = micros() - lastmicro;
  return buff;
}


void setup1() {
  //SERIAL_8E1 - 8 data bits, even parity, 1 stop bits
  //SERIAL_8N1 - 8 data bits, no parity, 1 stop bit (default)
  Serial1.setTX(16);
  Serial1.setRX(17);
  Serial1.begin(UART_BAUD_RATE);
  Serial1.setTimeout(UART_COMMAND_TIMEOUT_MS);
  //Serial1.begin(UART_BAUD_RATE, SERIAL_8E1, 17, 18);

  Serial.begin(UART_BAUD_RATE);
  Serial.setTimeout(UART_COMMAND_TIMEOUT_MS);
#if WAIT_FOR_USB_SERIAL
  while (!Serial)
    ;
#endif

#if (usbdebug)
  Serial.println(__FILE__);
#endif

  pinMode(LED1_pin, OUTPUT);
  pinMode(LED2_pin, OUTPUT);
  pinMode(LED3_pin, OUTPUT);
  digitalWrite(LED1_pin, 1);
  digitalWrite(LED2_pin, 0);
  digitalWrite(LED3_pin, 0);

#if (usbdebug)
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, 1);
#endif
}

static bool copy_coherent_measurements(int16_t *destination,
                                       uint32_t *generation_out) {
  // A sequence lock is much smaller than pausing ADC conversion, and protects
  // the six-word snapshot while the two RP2350 cores run independently.
  for (uint8_t attempt = 0; attempt < 8; attempt++) {
    const uint32_t before =
        __atomic_load_n(&tc_sample_generation, __ATOMIC_ACQUIRE);
    if (before == 0 || (before & 1U)) {
      continue;
    }

    memcpy(destination, tc_voltdata_int16t, sizeof(tc_voltdata_int16t));

    const uint32_t after =
        __atomic_load_n(&tc_sample_generation, __ATOMIC_ACQUIRE);
    if (before == after && !(after & 1U)) {
      if (generation_out != nullptr) {
        *generation_out = after;
      }
      return true;
    }
  }

  return false;
}

void copy_latest_measurements(int16_t *destination) {
  (void)copy_coherent_measurements(destination, nullptr);
}

static bool read_next_command_packet(Stream &port, byte *command_line) {
  if (port.available() <= 0) {
    return false;
  }

  // Keep the rest of a rate packet for consume_rate_set_packet_and_ack().
  if (port.peek() == RATE_SET_PACKET_START) {
    if (port.available() < COMMAND_PACKET_BYTES) {
      return false;
    }
    return port.readBytes(command_line, COMMAND_PACKET_BYTES) == COMMAND_PACKET_BYTES;
  }

  if (port.available() < COMMAND_PACKET_BYTES) {
    return false;
  }

  const int command = port.read();
  const int terminator = port.read();
  if (command < 0 || terminator != endofline) {
    return false;
  }

  command_line[0] = (byte)command;
  command_line[1] = (byte)terminator;
  return true;
}

void print_vi_channels(const int16_t *measurements) {
  Serial.print("V1=");
  Serial.print(measurements[0] / 5.0, 0);
  Serial.print(" I1=");
  Serial.print(measurements[1] / 5.0, 0);
  Serial.print(" | V2=");
  Serial.print(measurements[2] / 5.0, 0);
  Serial.print(" I2=");
  Serial.print(measurements[3] / 5.0, 0);
  Serial.print(" | V3=");
  Serial.print(measurements[4] / 5.0, 0);
  Serial.print(" I3=");
  Serial.println(measurements[5] / 5.0, 0);
}

static void append_feature_sample(const int16_t *measurements) {
  uint32_t head = __atomic_load_n(&feature_ring_head, __ATOMIC_RELAXED);
  uint32_t idx = head & (FEATURE_RING_SIZE - 1);

  for (int c = 0; c < transmit_data_channels; c++) {
    feature_ring_buffer[idx][c] = measurements[c];
  }

  __atomic_store_n(&feature_ring_head, head + 1, __ATOMIC_RELEASE);
}

static uint32_t snapshot_feature_window(float *fs_out) {
  uint32_t head_now = __atomic_load_n(&feature_ring_head, __ATOMIC_ACQUIRE);
  uint32_t now_us = micros();

  if (feature_fs_last_us == 0) {
    feature_fs_last_us = now_us;
    feature_fs_last_head = head_now;
  } else {
    uint32_t dt_us = now_us - feature_fs_last_us;
    if (dt_us >= 100000UL) {
      uint32_t dframes = head_now - feature_fs_last_head;
      if (dframes > 0) {
        feature_estimated_fs = ((float)dframes * 1000000.0f) / (float)dt_us;
      }
      feature_fs_last_us = now_us;
      feature_fs_last_head = head_now;
    }
  }

  uint32_t n = (head_now < FEATURE_WINDOW_SAMPLES) ? head_now : FEATURE_WINDOW_SAMPLES;
  if (n < 64) {
    *fs_out = feature_estimated_fs;
    return 0;
  }

  uint32_t start = head_now - n;
  for (uint32_t k = 0; k < n; k++) {
    uint32_t src = (start + k) & (FEATURE_RING_SIZE - 1);
    for (int c = 0; c < transmit_data_channels; c++) {
      feature_snapshot[k * transmit_data_channels + c] = feature_ring_buffer[src][c];
    }
  }

  *fs_out = feature_estimated_fs;
  return n;
}

static void print_feature_summary(const PowerMetrics *m) {
  Serial.print("[FEATURE] fs=");
  Serial.print(m->sample_rate_hz);
  Serial.print(" n=");
  Serial.print(m->n_samples);
  Serial.print(" freq=");
  Serial.print(m->frequency, 3);
  Serial.print("Hz");

  for (int phase = 0; phase < N_PHASES; phase++) {
    Serial.print(" | P");
    Serial.print(phase + 1);
    Serial.print(" Vrms=");
    Serial.print(m->Vrms[phase], 3);
    Serial.print(" Irms=");
    Serial.print(m->Irms[phase], 3);
    Serial.print(" P=");
    Serial.print(m->P[phase], 3);
    Serial.print(" S=");
    Serial.print(m->S[phase], 3);
    Serial.print(" PF=");
    Serial.print(m->PF[phase], 3);
  }

  Serial.println();
}

static void write_u16_le(uint8_t *dst, uint16_t value) {
  dst[0] = (uint8_t)(value & 0xFF);
  dst[1] = (uint8_t)((value >> 8) & 0xFF);
}

static void write_i16_le(uint8_t *dst, int16_t value) {
  write_u16_le(dst, (uint16_t)value);
}

static void send_raw_frame_to_esp(const int16_t *measurements) {
  uint8_t frame[EF_RAW_FRAME_BYTES];
  const uint16_t payload_len = EF_RAW_PAYLOAD_BYTES;

  frame[0] = EF_RAW_START_BYTE;
  frame[1] = EF_RAW_TYPE_SAMPLE;
  write_u16_le(&frame[2], payload_len);

  for (int ch = 0; ch < transmit_data_channels; ch++) {
    write_i16_le(&frame[4 + (2 * ch)], measurements[ch]);
  }

  const uint16_t crc = crc16_ccitt(&frame[2], 2 + payload_len);
  write_u16_le(&frame[4 + payload_len], crc);
  frame[EF_RAW_FRAME_BYTES - 1] = EF_RAW_END_BYTE;

  Serial1.write(frame, sizeof(frame));
}

static bool send_metrics_packet_to_esp(void) {
#if ENABLE_PI_FEATURE_COMPUTE
  if (!latest_feature_valid) {
    return false;
  }

  PowerMetrics metrics_copy = latest_feature_metrics;
  uint8_t packet[METRICS_PACKET_BYTES];

  packet[0] = METRICS_START_BYTE;
  packet[1] = METRICS_TYPE_DERIVED;

  const uint16_t payload_len = serialize_metrics(&metrics_copy, &packet[4]);
  if (payload_len != METRICS_PAYLOAD_BYTES) {
    return false;
  }

  write_u16_le(&packet[2], payload_len);

  const uint16_t crc = crc16_ccitt(&packet[4], payload_len);
  write_u16_le(&packet[4 + payload_len], crc);
  packet[METRICS_PACKET_BYTES - 1] = METRICS_END_BYTE;

  Serial1.write(packet, sizeof(packet));
  return true;
#else
  return false;
#endif
}

static bool consume_rate_set_packet_and_ack(bool from_serial1, uint8_t period_lsb) {
  uint8_t tail[RATE_PACKET_REMAINING_BYTES] = {0};
  size_t got = 0;

  if (from_serial1) {
    got = Serial1.readBytes(tail, sizeof(tail));
  } else {
    got = Serial.readBytes(tail, sizeof(tail));
  }

  if (got != sizeof(tail) || tail[RATE_PACKET_REMAINING_BYTES - 1] != RATE_SET_PACKET_END) {
    return false;
  }

  const uint8_t ack[RATE_ACK_BYTES] = {
    RATE_ACK_START,
    period_lsb,
    RATE_SET_PACKET_END
  };

  if (from_serial1) {
    Serial1.write(ack, sizeof(ack));
  } else {
    Serial.write(ack, sizeof(ack));
  }

  return true;
}

static int16_t median3_i16(int16_t a, int16_t b, int16_t c) {
  if (a > b) {
    int16_t t = a;
    a = b;
    b = t;
  }
  if (b > c) {
    int16_t t = b;
    b = c;
    c = t;
  }
  if (a > b) {
    b = a;
  }
  return b;
}

static void deglitch_feature_snapshot(uint32_t n) {
  if (n < 3) {
    return;
  }

  memcpy(feature_snapshot_filter,
         feature_snapshot,
         n * transmit_data_channels * sizeof(int16_t));

  for (uint32_t k = 1; k + 1 < n; k++) {
    for (int c = 0; c < transmit_data_channels; c++) {
      feature_snapshot[k * transmit_data_channels + c] =
          median3_i16(feature_snapshot_filter[(k - 1) * transmit_data_channels + c],
                      feature_snapshot_filter[k * transmit_data_channels + c],
                      feature_snapshot_filter[(k + 1) * transmit_data_channels + c]);
    }
  }
}

static void print_feature_raw_window_stats(uint32_t n) {
#if FEATURE_RAW_WINDOW_DEBUG
  static const char *names[transmit_data_channels] = {
    "V1", "I1", "V2", "I2", "V3", "I3"
  };
  int16_t min_v[transmit_data_channels];
  int16_t max_v[transmit_data_channels];
  int64_t sum_v[transmit_data_channels];
  double sum_sq_v[transmit_data_channels];

  for (int c = 0; c < transmit_data_channels; c++) {
    min_v[c] = 32767;
    max_v[c] = -32768;
    sum_v[c] = 0;
    sum_sq_v[c] = 0.0;
  }

  for (uint32_t k = 0; k < n; k++) {
    for (int c = 0; c < transmit_data_channels; c++) {
      int16_t value = feature_snapshot[k * transmit_data_channels + c];
      if (value < min_v[c]) min_v[c] = value;
      if (value > max_v[c]) max_v[c] = value;
      sum_v[c] += value;
      sum_sq_v[c] += (double)value * (double)value;
    }
  }

  Serial.print("[FEATURE_RAW]");
  for (int c = 0; c < transmit_data_channels; c++) {
    double mean = (double)sum_v[c] / (double)n;
    double variance = (sum_sq_v[c] / (double)n) - (mean * mean);
    if (variance < 0.0) variance = 0.0;
    double centered_rms = sqrt(variance);

    Serial.print(" ");
    Serial.print(names[c]);
    Serial.print("[");
    Serial.print(min_v[c] / 5.0, 0);
    Serial.print("..");
    Serial.print(max_v[c] / 5.0, 0);
    Serial.print("] rms=");
    Serial.print(centered_rms / 5.0, 1);
  }
  Serial.println();
#else
  (void)n;
#endif
}

static void compute_and_print_feature_if_due(void) {
  static uint32_t last_feature_ms = 0;
  uint32_t now_ms = millis();
  if ((uint32_t)(now_ms - last_feature_ms) < FEATURE_PRINT_INTERVAL_MS) {
    return;
  }
  last_feature_ms = now_ms;

  float fs = 0.0f;
  uint32_t n = snapshot_feature_window(&fs);
  if (n == 0 || fs < 1.0f) {
#if FEATURE_SERIAL_PRINT
    Serial.print("[FEATURE] waiting n=");
    Serial.print(n);
    Serial.print(" fs=");
    Serial.println(fs, 1);
#endif
    return;
  }

  deglitch_feature_snapshot(n);
#if FEATURE_SERIAL_PRINT
  print_feature_raw_window_stats(n);
#endif
  compute_metrics(feature_snapshot, n, fs, &latest_feature_metrics);
  latest_feature_metrics.timestamp_us = micros();
  latest_feature_metrics.box_temp_c = PI_BOX_TEMP_UNAVAILABLE_C;
  latest_feature_valid = true;

#if FEATURE_SERIAL_PRINT
  print_feature_summary(&latest_feature_metrics);
#endif
}

void loop1() {
  static int i;
  static byte command[1] = { 0 };
  static byte LED1bit = 0;
  static byte LED2bit = 0;
  static byte LED3bit = 0;
  static byte commandByte_line[COMMAND_PACKET_BYTES];
  static byte commandByte;
  static byte read_flag = 0;
  static byte metrics_flag = 0;
  static byte command_received_flag = 0;
  static byte command_from_serial1 = 0;
  static uint32_t last_sent_generation = 0;

  if (read_next_command_packet(Serial, commandByte_line)) {
    command_received_flag = 1;
    command_from_serial1 = 0;
#if (usbdebug)
    Serial.print("command from Serial0: ");
    //Serial.print(commandByte);
    Serial.print(" ");
#endif
  }

  if (read_next_command_packet(Serial1, commandByte_line)) {
    command_received_flag = 1;
    command_from_serial1 = 1;
#if (usbdebug)
    Serial.print("command from Serial1: ");
    //Serial.print(commandByte);
    Serial.print(" ");
#endif
  }

  if (command_received_flag) {
    command_received_flag = 0;

    if (commandByte_line[0] == RATE_SET_PACKET_START) {
      consume_rate_set_packet_and_ack(command_from_serial1, commandByte_line[1]);
      return;
    }

    // commandbyte format:
    // bit 7 (MSB) LED1 input
    // bit 6       LED2 input
    // bit 5       LED3 input
    // bit 4
    // bit 3       request data flag
    // bit 2       request derived metrics packet
    // bit 1
    // bit 0

    commandByte = commandByte_line[0];

    LED1bit = (commandByte >> 7) & 0b01;
    LED2bit = (commandByte >> 6) & 0b01;
    LED3bit = (commandByte >> 5) & 0b01;
    read_flag = (commandByte >> 3) & 0b01;
    metrics_flag = (commandByte >> 2) & 0b01;

    // Update forwarded LED state as soon as the command is accepted. Do not
    // let an ADC wait path hide ESP connectivity/state from the front panel.
    digitalWrite(LED1_pin, LED1bit);
    digitalWrite(LED2_pin, LED2bit);
    digitalWrite(LED3_pin, LED3bit);
#if (usbdebug)
    digitalWrite(LED_BUILTIN, 0);

    for (int j = 0; j < COMMAND_PACKET_BYTES; j++) {
      Serial.println(" ");
      for (i = 0; i < 8; i++) {
        Serial.print(((commandByte_line[j] >> (7 - i)) & 0x01));
      }
    }

    Serial.println("");
    Serial.print(" ");
    Serial.print("LEDs: ");
    Serial.print(LED1bit);
    Serial.print(LED2bit);
    Serial.print(LED3bit);
    Serial.print(" read flag: ");
    Serial.println(read_flag);
#endif

    if (!read_flag) {
      // user wishes to manipulate the LED and none else
    } else {
      // user wishes to read data


      // #if (usbdebug)
      //       Serial.println("waiting for data...");
      // #endif
#if (usbdebug)
      Serial.print("count:");
      Serial.println(count);
#endif
      count = 0;

      // One request consumes one fresh, coherent ADC snapshot. This preserves
      // the raw stream without replaying an already-sent sample.
      uint32_t raw_wait_start_us = micros();
      uint32_t generation = 0;
      bool copied = false;
      while ((uint32_t)(micros() - raw_wait_start_us) < RAW_DATAREADY_WAIT_TIMEOUT_US) {
        if (copy_coherent_measurements(voltdata_int16t_core1rxbuffer, &generation) &&
            generation != last_sent_generation) {
          copied = true;
          break;
        }
      }

      if (!copied) {
        // Do not fabricate a stale frame. The ESP32 will retry this request.
        return;
      }

      last_sent_generation = generation;


// reading is done -- ready to copy data to writebuffer
#if (usbdebug)  // print to serial instead of write to receiving controller
      Serial.print("v: ");
      for (i = 0; i < (sizeof(voltdata_int16t_core1rxbuffer) / sizeof(voltdata_int16t_core1rxbuffer[0])); i++) {
        Serial.print(voltdata_int16t_core1rxbuffer[i] / 5.0, 0);  // display in mV
        Serial.print(" ");
      }
      Serial.println("");
      Serial.print("mult: ");
      Serial.println(mult, 15);
#endif

      send_raw_frame_to_esp(voltdata_int16t_core1rxbuffer);

    }  // end if(!readflag)

    if (metrics_flag) {
#if ENABLE_PI_FEATURE_COMPUTE
      send_metrics_packet_to_esp();
#endif
    }

    digitalWrite(LED_BUILTIN, 1);
  }

#if RAW_SERIAL_PRINT
  if (Serial && tc_dataready_flag && ((millis() - previousMillis) >= interval)) {
    previousMillis = millis();
    copy_latest_measurements(voltdata_int16t_printbuffer);
    print_vi_channels(voltdata_int16t_printbuffer);
  }
#endif

#if ENABLE_PI_FEATURE_COMPUTE
  compute_and_print_feature_if_due();
#endif
}

void loop() {
  static bool firstrun = 1;
  static int i;

  if (firstrun) {
    mult = 2.0 * mcp.getReference() / (double)mcp.getMaxValue();
    mult = mult * 5000.0;
    firstrun = 0;
  }

  // รอ ADC scan ครบ 1 รอบ (IRQ ยิงเมื่อ scan เสร็จทุก channel)
  unsigned long wait_start = micros();
  while (!adc_ready) {
    if (micros() - wait_start > 10000) {
      return;
    }
  }
  adc_ready = false;

  // อ่านทีเดียวทุก channel ตามลำดับ V1 I1 V2 I2 V3 I3
  // CH1=V1, CH2=I1, CH3=V2, CH4=I2, CH5=V3, CH6=I3
  for (i = 0; i < transmit_data_channels; i++) {
    raw_read_bits[i] = mcp.analogReadScan(MCP_CH_arr[i + 1]);
  }

  voltdata_int16t_core0txbuffer[0] = ((double)raw_read_bits[0] * mult); // V1 ← CH1
  voltdata_int16t_core0txbuffer[1] = ((double)raw_read_bits[1] * mult); // I1 ← CH2
  voltdata_int16t_core0txbuffer[2] = ((double)raw_read_bits[2] * mult); // V2 ← CH3
  voltdata_int16t_core0txbuffer[3] = ((double)raw_read_bits[3] * mult); // I2 ← CH4
  voltdata_int16t_core0txbuffer[4] = ((double)raw_read_bits[4] * mult); // V3 ← CH5
  voltdata_int16t_core0txbuffer[5] = ((double)raw_read_bits[5] * mult); // I3 ← CH6

  for (i = 0; i < transmit_data_channels; i++) {
    if (voltdata_int16t_core0txbuffer[i] < 0) voltdata_int16t_core0txbuffer[i] = 0;
  }

  // publish ไป Core 1 เหมือนเดิม
  const uint32_t publishing_generation =
      __atomic_fetch_add(&tc_sample_generation, 1U, __ATOMIC_ACQ_REL) + 1U;
  memcpy(tc_voltdata_int16t, voltdata_int16t_core0txbuffer, sizeof(tc_voltdata_int16t));
  tc_has_sample = true;
  tc_dataready_flag = true;
  __atomic_store_n(&tc_sample_generation, publishing_generation + 1U, __ATOMIC_RELEASE);

#if ENABLE_PI_FEATURE_COMPUTE
  append_feature_sample(voltdata_int16t_core0txbuffer);
#endif
  count++;
}
