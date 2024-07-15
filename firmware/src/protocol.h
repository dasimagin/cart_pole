#pragma once
#include <pb_decode.h>
#include <pb_encode.h>

#include "CRC.h"
#include "CRC8.h"
#include "cobs.h"
#include "helpers.h"
#include "proto/protocol.pb.h"

namespace cartpole {

class Protocol {
private:
    static const uint32_t SERIAL_SPEED = 2000000;
    static const uint8_t FRAME_EOF = 0x00;

    uint8_t rxBuff[256];
    uint8_t txBuff[256];
    CRC8 crc;
    HardwareError error;

public:
    // TODO: Convert to setters?
    std::function<State()> resetCallback = nullptr;
    std::function<State(Target)> targetCallback = nullptr;
    std::function<Config(Config)> configCallback = nullptr;

    Protocol() = default;

    void init() {
        Serial.begin(SERIAL_SPEED);
        Serial.print("start");  // TODO: More concize startup message?
        CREATE_TASK("protocol", _pollingTask, this);
    }

    HardwareError getErrors() { return error; }

private:
    void _poll() {
        if (!Serial.available()) return;
        size_t length = Serial.readBytesUntil(FRAME_EOF, rxBuff, sizeof(rxBuff));
        if (!length) return;
        cobs::decode(rxBuff, length);
        uint8_t type = rxBuff[1];

        if (type == RequestType_RESET && resetCallback) {
            auto state = resetCallback();
            auto stream = _prepare_stream(RequestType_RESET);
            pb_encode(&stream, &State_msg, &state);
            _flush_stream(stream);
        }

        if (type == RequestType_TARGET && targetCallback) {
            auto target = _decode<Target>(Target_msg);
            auto state = targetCallback(target);
            auto stream = _prepare_stream(RequestType_TARGET);
            pb_encode(&stream, &State_msg, &state);
            _flush_stream(stream);
        }

        if (type == RequestType_CONFIG && configCallback) {
            auto config = _decode<Config>(Target_msg);
            config = configCallback(config);
            auto stream = _prepare_stream(RequestType_CONFIG);
            pb_encode(&stream, &Config_msg, &config);
            _flush_stream(stream);
        }
    }

    static void _pollingTask(void *params) {
        Protocol *protocol = (Protocol *)params;
        while (true) {
            protocol->_poll();
            RTOS_YIELD;
        }
    }

    template <typename T>
    T _decode(pb_msgdesc_t descriptor) {
        // TODO: Return error flag
        // TODO: Check CRC
        size_t payload_len = rxBuff[2];
        pb_istream_t stream = pb_istream_from_buffer(rxBuff + 3, payload_len);
        T message;
        bool ok = pb_decode(&stream, &descriptor, &message);
        if (!ok) error = HardwareError_PROTOCOL_DECODE_ERROR;
        return message;
    }

    pb_ostream_t _prepare_stream(RequestType type) {
        // [COBS BYTE] [TYPE] [LEN+1] [DATA] [CRC8] [EOF]
        rxBuff[0] = 0;     // Reserved for COBS
        rxBuff[1] = type;  // Payload type
        rxBuff[2] = 0;     // Payload length
        // Data region starts at byte 3
        // 5 extra bytes reserved for COBS, TYPE, LEN, CRC8, EOF
        return pb_ostream_from_buffer(rxBuff + 3, sizeof(rxBuff) - 5);
    }

    void _flush_stream(pb_ostream_t &stream) {
        size_t data_length = stream.bytes_written;
        rxBuff[2] = data_length;
        crc.restart();
        crc.add(rxBuff + 1, 2 + data_length);
        rxBuff[3 + data_length] = crc.getCRC();
        rxBuff[4 + data_length] = FRAME_EOF;
        cobs::encode(rxBuff, 4 + data_length);
        Serial.write(rxBuff, 5 + data_length);
    }
};

}  // namespace cartpole