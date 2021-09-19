#include "stepper.h"

#include "globals.h"
#include "protocol_processor.h"

#include <sstream>
#include <iomanip>


namespace {
    const int TMC_EN = 25;
    const int TMC_STEP = 26;
    const int TMC_DIR = 27;
    const int ENDSTOP_LEFT = 18;
    const int ENDSTOP_RIGHT = 19;

    const float STEPPER_CURRENT = 0.6;
    const int SERIAL_SPEED = 115200;
    const int ADDRESS = 0b00;
    const float R_SENSE = 0.11;
    const int TOFF_VALUE = 5;
    const int MICROSTEPS = 16;
    const bool REVERSE_STEPPER = true;
    const int FULL_STEPS_PER_METER = 5000;
    const float HOMING_SPEED = 0.1;
    const float HOMING_ACCELERATION = 0.5;

    const int METERS_TO_STEPS_MULTIPLIER = MICROSTEPS * FULL_STEPS_PER_METER;

    TaskHandle_t HOMING_TASK_HANDLE = nullptr;
    bool IS_DONE_HOMING = false;

    ProtocolProcessor &P = GetProtocolProcessor();
    Globals &G = GetGlobals();

    void homingTask(void *) {
        Stepper &S = GetStepper();
        S.Homing();
        
        IS_DONE_HOMING = true;
        vTaskDelete(HOMING_TASK_HANDLE);
        while(true) {}
    }
}

Stepper::Stepper() : tmc_serial_port(Serial2), tmc_driver(&tmc_serial_port, R_SENSE, ADDRESS) {
    pinMode(TMC_EN, OUTPUT);
    pinMode(TMC_STEP, OUTPUT);
    pinMode(TMC_DIR, OUTPUT);
    pinMode(ENDSTOP_LEFT, INPUT);
    pinMode(ENDSTOP_RIGHT, INPUT);

    digitalWrite(TMC_EN, LOW);
    delay(10);
    tmc_serial_port.begin(SERIAL_SPEED);
    tmc_driver.begin();
    tmc_driver.rms_current(STEPPER_CURRENT * 1000);
    tmc_driver.microsteps(MICROSTEPS);
    tmc_driver.toff(0);

    // Init FastAccelStepper
    fas_engine.init();
    fas_stepper = fas_engine.stepperConnectToPin(TMC_STEP);
    assert(fas_stepper != NULL);
    fas_stepper->setDirectionPin(TMC_DIR, REVERSE_STEPPER);
}

void Stepper::Enable() {
    tmc_driver.toff(TOFF_VALUE);
    P.Log("Stepper enabled");
}

void Stepper::Disable() {
    tmc_driver.toff(0);
    P.Log("Stepper disabled");
}

void Stepper::ForceStop() {
    fas_stepper->forceStopAndNewPosition(fas_stepper->getCurrentPosition());
    P.Log("Force stopped stepper");
}

float Stepper::GetCurrentPosition() {
    int pos_steps = fas_stepper->getCurrentPosition();
    return static_cast<float>(pos_steps) / METERS_TO_STEPS_MULTIPLIER;
}

void Stepper::Homing() {
    ForceStop();
    Enable();
    SetSpeed(HOMING_SPEED);
    SetAcceleration(HOMING_ACCELERATION);

    // RUN LEFT
    fas_stepper->runBackward();
    while (!digitalRead(ENDSTOP_LEFT)) {}

    ForceStop();
    fas_stepper->setCurrentPosition(0);
    delay(50);

    // RUN RIGHT
    fas_stepper->runForward();
    while (!digitalRead(ENDSTOP_RIGHT)) {}

    ForceStop();
    int delta_steps = fas_stepper->getCurrentPosition();
    fas_stepper->setCurrentPosition(delta_steps);
    delay(50);

    // GOTO CENTER
    fas_stepper->moveTo(delta_steps / 2);
    while (fas_stepper->isRunning()) {}

    G.full_length_meters = static_cast<float>(delta_steps) / METERS_TO_STEPS_MULTIPLIER;
    G.hw_max_x = G.full_length_meters / 2;
    
    // TODO: Reset state?
    G.errcode = Error::NO_ERROR;
    std::stringstream stream;
    stream << std::fixed << std::setprecision(5) << "Full length: " << 
        delta_steps << " steps" << G.full_length_meters << " meters";
    P.Log(stream.str());
    stream = std::stringstream{};
    stream << std::fixed << std::setprecision(5) << "Valid X range: " << - G.hw_max_x << " ... " << G.hw_max_x;
    P.Log(stream.str());
}

void Stepper::AsyncHoming() {
    IS_DONE_HOMING = false;
    BaseType_t ret = xTaskCreate(
        homingTask,
        "homing",
        8192,
        nullptr,
        tskIDLE_PRIORITY,
        &HOMING_TASK_HANDLE
    );

    if (ret != pdPASS) {
        P.Error("Async Homing Failure");
    }
    if (ret == errCOULD_NOT_ALLOCATE_REQUIRED_MEMORY) {
        P.Log("Could not allocate required memory");
    }
}

bool Stepper::IsDoneHoming() {
    return IS_DONE_HOMING;
}

void Stepper::SetSpeed(float value) {
    uint32_t speed_hz = static_cast<uint32_t>(value * METERS_TO_STEPS_MULTIPLIER);
    fas_stepper->setSpeedInHz(speed_hz);
}

void Stepper::SetAcceleration(float value) {
    uint32_t steps_per_ss = static_cast<uint32_t>(value * METERS_TO_STEPS_MULTIPLIER);
    fas_stepper->setAcceleration(steps_per_ss);

    std::stringstream stream;
    stream << std::fixed << std::setprecision(5) << "Set stepper acceleration: " <<
        value << " m/s^2, " << steps_per_ss << " steps/s^2";
    P.Log(stream.str());
}

void Stepper::SetTargetPosition(float value) {
    int pos_steps = static_cast<int>(value * METERS_TO_STEPS_MULTIPLIER);
    fas_stepper->moveTo(pos_steps);
}

Stepper &GetStepper() {
    static Stepper stepper{};
    return stepper;
}
