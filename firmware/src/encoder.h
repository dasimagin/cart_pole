#pragma once

#include <Arduino.h>

#include "driver/spi_master.h"
#include "helpers.h"
#include "proto/protocol.pb.h"

namespace cartpole {

// CRC-8, custom, poly 0x1D, initial 0xFF, final xor 0xFF
// Example from datasheet: 0x50 0x81 0x08 0x04 -> 0x89
// http://www.sunshine2k.de/coding/javascript/crc/crc_js.html
constexpr uint8_t CRC_TABLE[256] = {
    0x00, 0x1D, 0x3A, 0x27, 0x74, 0x69, 0x4E, 0x53, 0xE8, 0xF5, 0xD2, 0xCF, 0x9C, 0x81, 0xA6, 0xBB,
    0xCD, 0xD0, 0xF7, 0xEA, 0xB9, 0xA4, 0x83, 0x9E, 0x25, 0x38, 0x1F, 0x02, 0x51, 0x4C, 0x6B, 0x76,
    0x87, 0x9A, 0xBD, 0xA0, 0xF3, 0xEE, 0xC9, 0xD4, 0x6F, 0x72, 0x55, 0x48, 0x1B, 0x06, 0x21, 0x3C,
    0x4A, 0x57, 0x70, 0x6D, 0x3E, 0x23, 0x04, 0x19, 0xA2, 0xBF, 0x98, 0x85, 0xD6, 0xCB, 0xEC, 0xF1,
    0x13, 0x0E, 0x29, 0x34, 0x67, 0x7A, 0x5D, 0x40, 0xFB, 0xE6, 0xC1, 0xDC, 0x8F, 0x92, 0xB5, 0xA8,
    0xDE, 0xC3, 0xE4, 0xF9, 0xAA, 0xB7, 0x90, 0x8D, 0x36, 0x2B, 0x0C, 0x11, 0x42, 0x5F, 0x78, 0x65,
    0x94, 0x89, 0xAE, 0xB3, 0xE0, 0xFD, 0xDA, 0xC7, 0x7C, 0x61, 0x46, 0x5B, 0x08, 0x15, 0x32, 0x2F,
    0x59, 0x44, 0x63, 0x7E, 0x2D, 0x30, 0x17, 0x0A, 0xB1, 0xAC, 0x8B, 0x96, 0xC5, 0xD8, 0xFF, 0xE2,
    0x26, 0x3B, 0x1C, 0x01, 0x52, 0x4F, 0x68, 0x75, 0xCE, 0xD3, 0xF4, 0xE9, 0xBA, 0xA7, 0x80, 0x9D,
    0xEB, 0xF6, 0xD1, 0xCC, 0x9F, 0x82, 0xA5, 0xB8, 0x03, 0x1E, 0x39, 0x24, 0x77, 0x6A, 0x4D, 0x50,
    0xA1, 0xBC, 0x9B, 0x86, 0xD5, 0xC8, 0xEF, 0xF2, 0x49, 0x54, 0x73, 0x6E, 0x3D, 0x20, 0x07, 0x1A,
    0x6C, 0x71, 0x56, 0x4B, 0x18, 0x05, 0x22, 0x3F, 0x84, 0x99, 0xBE, 0xA3, 0xF0, 0xED, 0xCA, 0xD7,
    0x35, 0x28, 0x0F, 0x12, 0x41, 0x5C, 0x7B, 0x66, 0xDD, 0xC0, 0xE7, 0xFA, 0xA9, 0xB4, 0x93, 0x8E,
    0xF8, 0xE5, 0xC2, 0xDF, 0x8C, 0x91, 0xB6, 0xAB, 0x10, 0x0D, 0x2A, 0x37, 0x64, 0x79, 0x5E, 0x43,
    0xB2, 0xAF, 0x88, 0x95, 0xC6, 0xDB, 0xFC, 0xE1, 0x5A, 0x47, 0x60, 0x7D, 0x2E, 0x33, 0x14, 0x09,
    0x7F, 0x62, 0x45, 0x58, 0x0B, 0x16, 0x31, 0x2C, 0x97, 0x8A, 0xAD, 0xB0, 0xE3, 0xFE, 0xD9, 0xC4,
};

class Encoder {
public:
    static const auto SPI_HOST = HSPI_HOST;
    static const int SPI_SPEED_HZ = 2000000;
    static const int MOSI_PIN = 13;
    static const int SCK_PIN = 14;
    static const int CS_PIN = 15;

    static constexpr float VELOCITY_FILTER_COEF = 0.8;  // larger = more smoothing
    static constexpr float ROTATION_THRESHOLD = 1.8 * PI;

    static const uint8_t REG_STAT = 0x00;
    static const uint8_t REG_ACSTAT = 0x01;
    static const uint8_t REG_AVAL = 0x02;
    static const uint8_t REG_ASPD = 0x03;
    static const uint8_t REG_MOD_1 = 0x06;
    static const uint8_t REG_MOD_2 = 0x08;

    static const uint16_t CHECK_BIT_14 = 0x4000;
    static const uint16_t UINT_TO_INT_15 = 0x8000;
    static constexpr float POW_2_15 = 32768.0;

public:
    spi_bus_config_t buscfg;
    spi_device_interface_config_t devcfg;
    spi_device_handle_t spi;

    unsigned long prev_time;
    float prev_raw_angle;
    int prev_rotations;
    float prev_angle;
    float prev_velocity;

public:
    Encoder() {
        buscfg = spi_bus_config_t{
            .mosi_io_num = MOSI_PIN,
            .miso_io_num = -1,
            .sclk_io_num = SCK_PIN,
        };
        devcfg = spi_device_interface_config_t{
            .mode = 1,
            .cs_ena_pretrans = 1,
            .clock_speed_hz = SPI_SPEED_HZ,
            .spics_io_num = CS_PIN,
            .flags = SPI_DEVICE_3WIRE | SPI_DEVICE_HALFDUPLEX,
            .queue_size = 1,
        };
    }

    void init() {
        // TODO: Move bus init outside of encoder class?
        ESP_ERROR_CHECK(spi_bus_initialize(SPI_HOST, &buscfg, SPI_DMA_DISABLED));
        ESP_ERROR_CHECK(spi_bus_add_device(SPI_HOST, &devcfg, &spi));
        reset();
        CREATE_TASK("encoder", _pollingTask, this);
    }

    void reset() {
        // Trigger hardware reset
        writeOneRegister(REG_ACSTAT, 0, 0, 1);
        delay(5);
        // Set update rate (FIR_MD) to 170.6 μs
        writeOneRegister(REG_MOD_1, 15, 14, 0b11);
        // Check status (all flags from NO_GMR_A to S_WD should be 0)
        uint16_t value = readOneRegister(REG_STAT);
        assert(getbits(value, 12, 1) == 0);
    }

    void update() {
        float curr_raw_angle = readRawAngle();
        unsigned long curr_time = micros();
        int curr_rotations = prev_rotations;

        if (abs(curr_raw_angle - prev_raw_angle) > ROTATION_THRESHOLD) {
            curr_rotations += (curr_raw_angle < prev_raw_angle) ? 1 : -1;
        }
        float curr_angle = curr_raw_angle + curr_rotations * 2.0f * M_PI;

        // https://docs.simplefoc.com/low_pass_filter
        float dt = (curr_time - prev_time) * 1e-6f;
        if (dt < 0.0f || dt > 0.5f) dt = 1e-3f;  // micros overflow
        float curr_raw_velocity = (curr_angle - prev_angle) / dt;
        float curr_velocity = VELOCITY_FILTER_COEF * prev_velocity +
                              (1.0f - VELOCITY_FILTER_COEF) * curr_raw_velocity;

        prev_time = curr_time;
        prev_raw_angle = curr_raw_angle;
        prev_rotations = curr_rotations;
        prev_angle = curr_angle;
        prev_velocity = curr_velocity;
    }

    float getAngle() { return prev_angle; }

    float getVelocity() { return prev_velocity; }

    HardwareError getErrors() { return HardwareError::HardwareError_NO_ERRORS; }

    // =================================================================

    static void _pollingTask(void *params) {
        Encoder *encoder = (Encoder *)params;
        while (true) {
            if (!encoder->getErrors()) encoder->update();
            vTaskDelay(1 / portTICK_PERIOD_MS);
        }
    }

    float readRawAngle() {
        uint16_t value = readOneRegister(REG_AVAL);
        value = getbits(value, 14, 0);  // drop 15th bit (RD_AV)
        if (value & CHECK_BIT_14) value -= UINT_TO_INT_15;
        return (2 * M_PI / POW_2_15) * ((float)(int16_t)value);
    }

    static inline void setbits(uint16_t &data, uint8_t start, uint8_t end, uint16_t value) {
        uint8_t len = start - end + 1;
        uint16_t mask = ((1 << len) - 1) << end;
        data = (data & ~mask) | (mask & (value << end));
    }

    static inline uint16_t getbits(uint16_t data, uint8_t start, uint8_t end) {
        uint8_t len = start - end + 1;
        uint16_t mask = ((1 << len) - 1) << end;
        return (data & mask) >> end;
    }

    void transcieve(uint8_t *txdata, uint8_t *rxdata, size_t txwords, size_t rxwords) {
        spi_transaction_t trans;
        memset(&trans, 0, sizeof(trans));
        trans.length = txwords * 16;
        trans.rxlength = rxwords * 16;
        trans.tx_buffer = txdata;
        trans.rx_buffer = rxdata;

        spi_device_acquire_bus(spi, portMAX_DELAY);
        ESP_ERROR_CHECK(spi_device_transmit(spi, &trans));
        spi_device_release_bus(spi);

        uint16_t safety_word = (rxdata[rxwords * 2 - 2] << 8) | rxdata[rxwords * 2 - 1];
        uint8_t crc = getCRC(txdata, txwords * 2);
        if (rxwords > 1) crc = getCRC(rxdata, rxwords * 2 - 2, ~crc);
        assert(crc == ((uint8_t)safety_word));
        assert(getbits(safety_word, 14, 12) == 0b111);
    }

    uint8_t getCRC(const uint8_t *data, size_t len, uint8_t seed = 0xFF) {
        uint8_t crc = seed;
        for (size_t i = 0; i < len; ++i) {
            crc = CRC_TABLE[crc ^ data[i]];
        }
        return crc ^ 0xFF;
    }

    uint16_t readOneRegister(uint8_t addr) {
        uint16_t request = 0;
        setbits(request, 15, 15, 1);   // rw
        setbits(request, 9, 4, addr);  // addr
        setbits(request, 3, 0, 1);     // nd

        uint8_t txdata[2];
        uint8_t rxdata[4];
        txdata[0] = request >> 8;
        txdata[1] = request;

        transcieve(txdata, rxdata, 1, 2);
        uint16_t value = (rxdata[0] << 8) | rxdata[1];
        return value;
    }

    void writeOneRegister(uint8_t addr, uint8_t start, uint8_t end, uint16_t value) {
        uint16_t curr_value = readOneRegister(addr);
        uint16_t new_value = curr_value;
        setbits(new_value, start, end, value);

        uint16_t request = 0;
        if (addr >= 0x05 && addr <= 0x11) {
            setbits(request, 14, 11, 0b1010);  // lock
        }
        setbits(request, 9, 4, addr);  // addr
        setbits(request, 3, 0, 1);     // nd

        uint8_t txdata[4];
        uint8_t rxdata[2];
        txdata[0] = request >> 8;
        txdata[1] = request;
        txdata[2] = new_value >> 8;
        txdata[3] = new_value;

        transcieve(txdata, rxdata, 2, 1);
    }
};
}  // namespace cartpole
