# PowerLens — Pi Pico 2W Firmware
### คู่มือสำหรับวิศวกร

---

## สารบัญ
1. [ภาพรวมระบบ](#1-ภาพรวมระบบ)
2. [Hardware ที่ใช้](#2-hardware-ที่ใช้)
3. [โครงสร้างไฟล์](#3-โครงสร้างไฟล์)
4. [การทำงานของ Dual-Core](#4-การทำงานของ-dual-core)
5. [Signal Chain — V และ I ตั้งแต่ ADC ถึง Output](#5-signal-chain--v-และ-i-ตั้งแต่-adc-ถึง-output)
6. [UART Protocol](#6-uart-protocol)
7. [การ Calibration](#7-การ-calibration)
8. [จุดที่ต้องแก้ไข — แก้ตรงไหน](#8-จุดที่ต้องแก้ไข--แก้ตรงไหน)
9. [Known Issues & ข้อควรระวัง](#9-known-issues--ข้อควรระวัง)

---

## 1. ภาพรวมระบบ

```
[MCP3564 ADC] ──SPI──► [Pi Pico 2W Core 0] ──Ring Buffer──► [Core 1] ──UART──► [ESP32]
                              ↑                                    ↑
                         อ่าน V, I 6 ช่อง                  คำนวณ metrics
                         800–2000 Hz                         ทุก 1 วินาที

[DS18B20] ──1-Wire──► [Core 1] ── อุณหภูมิกล่อง ──► [metrics packet]
```

ระบบวัดไฟฟ้า 3 เฟส (V1,I1 / V2,I2 / V3,I3) ผ่าน ADC แบบต่อเนื่อง  
Core 0 อ่านข้อมูลและใส่ ring buffer  
Core 1 รับคำสั่งจาก ESP32 และคำนวณ metric ทุก 1 วินาที

---

## 2. Hardware ที่ใช้

| ชิ้นส่วน | รุ่น / Spec | หมายเหตุ |
|---|---|---|
| MCU | Raspberry Pi Pico 2W | RP2350, dual-core Cortex-M33 |
| ADC | MCP3564 | 24-bit, 8-ch, SPI |
| CT clamp | — | ดู [Section 7](#7-การ-calibration) |
| Temp sensor | DS18B20 | 1-Wire, GPIO 22 |
| UART to ESP32 | Serial1 TX=GPIO16, RX=GPIO17 | 2 Mbaud |

**Pin map (Pi Pico 2W):**
```
GPIO 1  = MCLK (ไม่ใช้งาน)
GPIO 2  = IRQ  (ADC interrupt)
GPIO 3  = MOSI
GPIO 4  = MISO
GPIO 5  = CS
GPIO 6  = CLK
GPIO 8  = SPI Reset (ไม่ใช้งาน)
GPIO 13 = LED1
GPIO 14 = LED2
GPIO 15 = LED3
GPIO 16 = UART TX → ESP32
GPIO 17 = UART RX ← ESP32
GPIO 20 = SDA (สำรอง)
GPIO 21 = SCL (สำรอง)
GPIO 22 = DS18B20 Data
```

**Channel mapping ADC → Buffer:**
```
MCP3564 CH1 → raw_read_bits[1] → buffer[0] = V1
MCP3564 CH4 → raw_read_bits[4] → buffer[1] = I1
MCP3564 CH2 → raw_read_bits[2] → buffer[2] = V2
MCP3564 CH5 → raw_read_bits[5] → buffer[3] = I2
MCP3564 CH3 → raw_read_bits[3] → buffer[4] = V3
MCP3564 CH6 → raw_read_bits[6] → buffer[5] = I3
```

---

## 3. โครงสร้างไฟล์

```
test_program_uart_comm/
├── test_program_uart_comm.ino   ← จุดเริ่มต้นหลัก, Core 0 + Core 1 loop
├── power_metrics.h              ← struct PowerMetrics + calibration constants
├── power_metrics.cpp            ← อัลกอริทึมคำนวณ (Goertzel, Vrms, Irms, THD)
├── box_temp.h / .cpp            ← DS18B20 non-blocking driver
├── MCP3x6x_mod.h / .cpp        ← ADC driver (แก้จาก library ต้นฉบับ)
└── README.md                    ← ไฟล์นี้
```

---

## 4. การทำงานของ Dual-Core

### Core 0 — ADC Sampling (`loop()`)
```
ทุก frame (default 1000 µs = 1 kHz):
  1. อ่าน 7 ช่อง ADC ตามลำดับ CH0..CH6
  2. แปลง raw_ADC → int16:  int16 = raw_ADC × mult
     mult = 2 × Vref / 2²³ × 5000  →  int16/5000 = Vadc (volts)
  3. เขียนลง shared snapshot (seqlock) → Core 1 อ่านได้
  4. เขียนลง ring buffer (lock-free, SPSC)
  5. รอให้ครบ frame period (deadline-based timing)
```

### Core 1 — Command Handler + Compute (`loop1()`)
```
ทุก loop iteration (ไม่มี delay):
  1. ทุก 1000 ms → do_compute_metrics()
  2. BoxTemp::tick()           ← อ่านอุณหภูมิแบบ non-blocking
  3. รับคำสั่ง UART (Serial1 หรือ USB Serial)
     - 0xFC packet → เปลี่ยน sample rate
     - legacy 2-byte → LED control + data request
```

### Shared Data (ระหว่าง 2 Core)
```
ring_buffer[2048][6]    ← Core 0 เขียน, Core 1 อ่าน (lock-free ด้วย ring_head)
tc_voltdata_int16t[6]   ← Core 0 เขียน, Core 1 อ่าน (seqlock ด้วย tc_snapshot_sequence)
tc_dataready_flag       ← Core 0 set, Core 1 clear
target_frame_period_us  ← Core 1 set, Core 0 อ่าน
rate_change_pending     ← Core 1 set, Core 0 clear
```

---

## 5. Signal Chain — V และ I ตั้งแต่ ADC ถึง Output

### 5.1 ADC → int16 (ใน `.ino`)

```
raw_ADC (int32, 24-bit signed)
  ↓ × mult
int16 = raw_ADC × (2 × 3.3 / 8,388,608 × 5000)
      = raw_ADC × 0.003934

ความหมาย: int16 / 5000 = Vadc (หน่วย Volt)
ตัวอย่าง: int16 = 8384 → Vadc = 1.677 V
```

> **หมายเหตุ**: ตัวประกอบ `2×` มาจาก MCP3564 เป็น bipolar ADC (±Vref)  
> ค่า `5000` เป็น scaling constant ไม่ได้หมายถึงอะไรทางฟิสิกส์

---

### 5.2 int16 → Vrms (ใน `power_metrics.cpp`)

```
vv[k] = int16[k] × V_SCALE          ← แปลงเป็น Volt

v_mean = mean(vv)                    ← คำนวณ DC bias
vv[k] -= v_mean                      ← ลบ DC ออก

Vrms = sqrt( mean( vv² ) )
     = AC_rms_count × V_SCALE

ตัวอย่าง: AC_rms_count = 2965, V_SCALE = 0.07420 → Vrms = 220 V
```

**V_SCALE คำนวณอย่างไร:**
```c
// power_metrics.h
V_SCALE = V_SCALE_BASE × (V_REF_ACTUAL / V_REF_OBSERVED)

// หมายความว่า:
// V_REF_ACTUAL   = แรงดันจริงที่มิเตอร์อ่านได้ (V)
// V_REF_OBSERVED = แรงดันที่ Pi คำนวณได้ด้วย V_SCALE_BASE
// new_scale = old_scale × (จริง / Pi_คำนวณ)
```

---

### 5.3 int16 → Irms (ใน `power_metrics.cpp`)

มี 2 โหมด เลือกด้วย `#define I_CT_NONLINEAR_ENABLE`

#### โหมด Linear (I_CT_NONLINEAR_ENABLE = 0) ← โหมดปัจจุบัน
```
ii[k] = int16[k] × I_SCALE          ← แปลงเป็น Ampere โดยตรง

i_mean = mean(ii)                    ← DC bias ของ CT output
ii[k] -= i_mean                      ← ลบ DC ออก (auto หรือ manual)

Irms = sqrt( mean( ii² ) )
     = AC_rms_count × I_SCALE
```

#### โหมด Nonlinear (I_CT_NONLINEAR_ENABLE = 1)
```
Vadc_ct[k] = int16[k] / 5000        ← แปลงกลับเป็น Volt

Vpp = max(Vadc_ct) - min(Vadc_ct)   ← วัด peak-to-peak ทั้ง window
formula_v = Vpp - I_CT_VPP_ZERO_OFFSET   ← หัก noise floor ออก

target_Irms = I_CT_K × formula_v ^ I_CT_EXP  ← curve จาก oscilloscope

waveform rescaled → rms = target_Irms
```

> **ข้อควรระวัง Nonlinear mode**: ถ้า `formula_v < 0` (Vpp น้อยกว่า noise floor)  
> ฟังก์ชันคืนค่า 0A และ Irms = 0 ทั้งหมด — ตรวจสอบ `I_CT_VPP_ZERO_OFFSET` ให้ถูกต้อง

---

### 5.4 การคำนวณ P, S, PF, Phase, THD

```
P   = mean(v[k] × i[k + shift])     ← Active power พร้อม phase compensation
S   = Vrms × Irms                    ← Apparent power
PF  = P / S  (clamp ±1)             ← Power factor

Fundamental (50 Hz) → Goertzel algorithm ← เร็วกว่า FFT สำหรับ specific bins
THD = sqrt(Σ harmonic²) / fundamental   ← 2nd–7th harmonic

Frequency → hysteresis zero-crossing บน V1 (ป้องกัน over-count บนสัญญาณ distorted)
```

---

## 6. UART Protocol

### ESP32 → Pi Pico

#### 6.1 Legacy Command (2 bytes)
```
Byte 0: [LED1][LED2][LED3][0][RAW][METRICS][0][0]
Byte 1: 0x0D (end of line)

bit 7 = LED1 output
bit 6 = LED2 output
bit 5 = LED3 output
bit 3 = request RAW snapshot
bit 2 = request DERIVED metrics packet
```

#### 6.2 Set Rate Packet (7 bytes)
```
[0xFC][period_us LE 4B][CRC8][0xFD]

period_us: ช่วงเวลาระหว่าง frame ใน microseconds
  MIN = 500 µs  (2 kHz)
  MAX = 50000 µs (20 Hz)
  DEFAULT = 1000 µs (1 kHz)

CRC8: poly=0x07, init=0x00 คำนวณจาก 4 bytes ของ period_us
```

### Pi Pico → ESP32

#### 6.3 RAW Snapshot (13 bytes)
```
[int16_V1_LO][int16_V1_HI] × 6 channels + [0x0D]
= 12 bytes data + 1 byte terminator
```

#### 6.4 DERIVED Metrics Packet (167 bytes)
```
[0xAA][0x02][LEN_LO][LEN_HI][payload 160B][CRC16_LO][CRC16_HI][0x55]

Payload (160 bytes, little-endian float):
  Vrms[3], Irms[3], P[3], S[3], PF[3]          ← 5 × 3 × 4 = 60 bytes
  amp_V[3], amp_I[3], angle_V[3], angle_I[3]    ← 4 × 3 × 4 = 48 bytes
  phase_diff[3], thd_V[3], thd_I[3]             ← 3 × 3 × 4 = 36 bytes
  frequency (float)                              ← 4 bytes
  box_temp_c (float, -127 = no sensor)           ← 4 bytes
  timestamp_us (uint32)                          ← 4 bytes
  n_samples (uint16), sample_rate_hz (uint16)    ← 4 bytes
                                                 = 160 bytes total

CRC16: CCITT poly=0x1021, init=0xFFFF, no reflection
```

---

## 7. การ Calibration

Calibration constants ทั้งหมดอยู่ใน **`power_metrics.h`**

### 7.1 Voltage Scale (V_SCALE)

```c
#define V_SCALE_BASE_PHASEx   0.0785904165665054f  // ค่าเริ่มต้น bench

// Field calibration:
// 1) อ่าน Vrms จากมิเตอร์อ้างอิง → V_REF_ACTUAL
// 2) อ่าน Vrms ที่ Pi คำนวณได้  → V_REF_OBSERVED
// 3) อัปเดตค่าด้านล่าง:
#define V_REF_ACTUAL_PHASEx   220.0f   // ← ค่าจากมิเตอร์ (V)
#define V_REF_OBSERVED_PHASEx 233.0f   // ← ค่าที่ Pi คำนวณได้ (V)
```

> ค่า V_SCALE_BASE สามารถตั้งตรงได้เลยถ้าไม่ต้องการ reference framework

### 7.2 Current Scale (I_SCALE)

```c
#define I_SCALE_BASE_PHASEx   0.000926820893136259f  // ค่าเริ่มต้น

// Field calibration:
// 1) วัด Irms จากมิเตอร์อ้างอิง → I_REF_ACTUAL
// 2) อ่าน Irms ที่ Pi คำนวณได้  → I_REF_OBSERVED (ด้วย BASE เท่านั้น)
// 3) อัปเดตค่าด้านล่าง:
#define I_REF_ACTUAL_PHASEx   3.64f    // ← ค่าจากมิเตอร์ (A)
#define I_REF_OBSERVED_PHASEx 0.937f   // ← ค่าที่ Pi คำนวณได้ด้วย BASE (A)
```

**วิธีคำนวณ I_REF_OBSERVED:**
```
ขั้นตอน:
1. ตั้ง I_REF_ACTUAL = I_REF_OBSERVED = 1.0 (ยกเลิก correction ชั่วคราว)
2. รัน firmware และอ่านค่า Irms ที่ Pi คำนวณได้ในขณะที่โหลดกระแสคงที่
3. บันทึกค่านั้นเป็น I_REF_OBSERVED
4. ตั้ง I_REF_ACTUAL = ค่าจากมิเตอร์อ้างอิง
5. รีแฟลช
```

### 7.3 DC Offset ของ CT (I_OFFSET)

```c
// ใช้เฉพาะเมื่อ I_CT_AUTO_ZERO_OFFSET = 0
#define I_OFFSET_V_PHASEx   0.0f   // หน่วย Volt (= int16_mean / 5000)

// ค่าที่วัดได้จากข้อมูล:
// Phase 1: 1.6547 V  (int16 mean ≈ 8274)
// Phase 2: 1.6546 V  (int16 mean ≈ 8273)
// Phase 3: 1.6292 V  (int16 mean ≈ 8146)
```

> **แนะนำ**: คงไว้ที่ `I_CT_AUTO_ZERO_OFFSET = 1` (ค่า default)  
> ระบบจะลบ mean อัตโนมัติทุก window ซึ่งดีกว่าใช้ค่าคงที่  
> เพราะ CT offset เปลี่ยนตามอุณหภูมิ

### 7.4 Nonlinear CT (สำหรับเปิดใช้ในอนาคต)

```c
#define I_CT_NONLINEAR_ENABLE   0   // 1 = เปิด, 0 = ปิด (linear)

// ถ้าเปิด — ต้องวัด curve จากฮาร์ดแวร์จริง:
#define I_CT_K                  650.7176f   // coefficient
#define I_CT_EXP                1.2f        // exponent
#define I_CT_VPP_ZERO_OFFSET_V_PHASEx  1.67f  // Vpp ตอนไม่มีกระแส (ต้องวัดใหม่)

// วิธีวัด I_CT_VPP_ZERO_OFFSET:
// 1. ติดตั้ง CT clamp บนสาย แต่อย่าเปิดโหลด
// 2. บันทึก raw int16 ของ channel I ไว้ 5 วินาที
// 3. คำนวณ Vpp = (max - min) / 5000  (หน่วย Volt)
// 4. ใช้ค่านั้นเป็น I_CT_VPP_ZERO_OFFSET
```

### 7.5 Phase Compensation

```c
// MCP3564 อ่านช่อง V ก่อน I ในทุก scan cycle
// ที่ 1 kHz: V นำหน้า I ประมาณ 0.6 ms ≈ 10.8° ที่ 50 Hz
#define V_I_PHASE_COMP_DEG   10.8f   // ปรับโดยวัดกับโหลด resistive (cos φ = 1)
```

---

## 8. จุดที่ต้องแก้ไข — แก้ตรงไหน

### ต้องการเปลี่ยน Sample Rate

```c
// test_program_uart_comm.ino บรรทัด ~167
#define DEFAULT_FRAME_PERIOD_US  1000   // 1 kHz  ← แก้ตรงนี้
#define MIN_FRAME_PERIOD_US       500   // 2 kHz max
#define MAX_FRAME_PERIOD_US     50000   // 20 Hz min
// หรือส่ง Set Rate Packet จาก ESP32 เพื่อเปลี่ยน runtime
```

### ต้องการเปลี่ยน Calibration (V หรือ I)

```c
// power_metrics.h
// แก้ค่า *_REF_ACTUAL_* และ *_REF_OBSERVED_* ตาม Section 7
// ไม่ต้องแก้ส่วนอื่น
```

### ต้องการเพิ่ม/ลด Harmonic

```c
// power_metrics.h
#define N_HARMONICS  7   // คำนวณ 2nd–7th harmonic
// ลดเพื่อประหยัด CPU, เพิ่มเพื่อ THD แม่นขึ้น
// ต้องแน่ใจว่า N_HARMONICS × 50 Hz < fs/2 (Nyquist)
```

### ต้องการเปลี่ยน GPIO Pin ของ DS18B20

```c
// test_program_uart_comm.ino
#define BOX_TEMP_PIN  22   // ← แก้เป็น GPIO ที่ต้องการ
```

### ต้องการเปลี่ยน UART Baud Rate

```c
// test_program_uart_comm.ino
#define UART_BAUD_RATE 2000000   // ← แก้ทั้งฝั่ง Pi และ ESP32
```

### ต้องการเปลี่ยน Compute Interval (ปัจจุบัน 1 Hz)

```c
// test_program_uart_comm.ino ใน loop1()
if ((uint32_t)(now_ms - last_compute_ms) >= 1000UL) {   // ← แก้ 1000 เป็นค่าที่ต้องการ (ms)
```

### ต้องการเปิด Debug Log ผ่าน USB Serial

```c
// test_program_uart_comm.ino
#define usbdebug 0   // ← เปลี่ยนเป็น 1
// จะพิมพ์ command bytes, raw values, และ metric ออก USB Serial
// คำเตือน: จะทำให้ Core 1 ช้าลงเล็กน้อย
```

### ต้องการเปลี่ยน Frequency ของกริด (ถ้าไม่ใช่ 50 Hz)

```c
// power_metrics.h
#define LINE_FREQ_HZ  50.0f   // ← เปลี่ยนเป็น 60.0f สำหรับ 60 Hz grid
```

---

## 9. Known Issues & ข้อควรระวัง

### 9.1 Race Condition — `tc_dataready_flag`
```
ความเสี่ยง: Compiler อาจ reorder การเขียน tc_dataready_flag
            ขึ้นมาก่อนที่ seqlock จะ commit เสร็จ
แนะนำแก้:  เปลี่ยนบรรทัด ~500 ใน .ino เป็น:
            __atomic_store_n(&tc_dataready_flag, true, __ATOMIC_RELEASE);
```

### 9.2 `mult` เป็น `volatile double` (64-bit ไม่ Atomic บน ARM)
```
ความเสี่ยง: Core 1 อาจอ่านค่า mult แบบ torn ในโหมด debug
ผลกระทบ:   เฉพาะ debug print (usbdebug=1) ไม่กระทบ production
```

### 9.3 Noise Floor บน Channel I
```
ถ้า CT clamp ติดตั้งใกล้สายไฟแรงดัน จะเกิด inductive coupling
→ Channel I มี AC noise สูง แม้ไม่มีกระแสไหล
→ Irms จะไม่เป็น 0 เมื่อ load = 0A
แก้: เดิน cable CT ให้ห่างจากสายไฟแรงดัน ≥ 10 cm
     หรือใช้ shielded cable สำหรับ CT output
```

### 9.4 Int16 Saturation ที่กระแสสูง
```
int16 saturate ที่ ±32767 counts = Vadc ≈ ±6.55 V
ถ้า CT output swing เกิน 3.3V (ADC rail) → clipping → Irms ผิด
ตรวจสอบ: ดู raw int16 ว่ามีค่าติด 32767 หรือ -32768 บ่อยไหม
```

### 9.5 Nonlinear Mode: formula_v < 0 → Irms = 0
```
ถ้า I_CT_VPP_ZERO_OFFSET สูงกว่า Vpp จริงของ CT
→ ct_vpp_to_current_rms_a() คืน 0A
→ current_scale = 0 → i_buf = 0 → Irms = 0A
ตรวจสอบ: วัด Vpp จริงตอน no-load แล้วใส่ใน I_CT_VPP_ZERO_OFFSET
```

### 9.6 Metrics Compute หยุดชะงักถ้า sample rate ต่ำเกิน
```c
// power_metrics.h
#define MIN_METRICS_SAMPLE_RATE_HZ  ((2.0f * LINE_FREQ_HZ * N_HARMONICS) + 1.0f)
// = 2 × 50 × 7 + 1 = 701 Hz
// ถ้า sample rate < 701 Hz → publishUnavailableMetrics() (n_samples = 0)
```

---

## สรุป Flow การแก้ไขที่พบบ่อย

```
ค่า Vrms ผิด      → แก้ V_REF_ACTUAL / V_REF_OBSERVED ใน power_metrics.h
ค่า Irms ผิด      → แก้ I_REF_ACTUAL / I_REF_OBSERVED ใน power_metrics.h
PF / Phase ผิด    → แก้ V_I_PHASE_COMP_DEG ใน power_metrics.h
Irms ไม่เป็น 0    → ตรวจ noise บน CT cable (ฮาร์ดแวร์) หรือเปิด I_CT_NONLINEAR_ENABLE
Sample rate ช้า   → แก้ DEFAULT_FRAME_PERIOD_US หรือส่ง Set Rate Packet
ไม่รับ UART       → ตรวจ UART_BAUD_RATE ให้ตรงกับ ESP32
DS18B20 อ่านไม่ได้ → ตรวจ BOX_TEMP_PIN และ pull-up 4.7kΩ ต่อ 3.3V
```

---

*Last updated: 2026-06-07 | Firmware version: MODIFIED (dual-core + ring buffer + metrics)*
