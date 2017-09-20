#include "MPU9250.h"
#include "classifier_detector.h"

MPU9250 myIMU;

int16_t raw[buff_size_move][3] = {}; // buffer to store acceleration data for movement classification
const uint8_t INTERVAL = 75;         // new acceleration data is sampled every 75ms
bool first_round = true;

int count_press = 0;                 // count of the number of classified press up
int count_sit = 0;                   // count of the number of classified sit up
int count_lunge = 0;                 // count of the number of classified lunge

void setup() {
  Wire.begin();
  // TWBR = 12;  // 400 kbit/sec I2C speed
  Serial.begin(38400);

  // Read the WHO_AM_I register, this is a good test of communication
  byte c = myIMU.readByte(MPU9250_ADDRESS, WHO_AM_I_MPU9250);
  Serial.print("MPU9250 "); Serial.print("I AM "); Serial.print(c, HEX);
  Serial.print(" I should be "); Serial.println(0x71, HEX);


  if (c == 0x71) // WHO_AM_I should always be 0x68
  {
    Serial.println("MPU9250 is online...");

    // Start by performing self test and reporting values
    myIMU.MPU9250SelfTest(myIMU.SelfTest);
    Serial.print("x-axis self test: acceleration trim within : ");
    Serial.print(myIMU.SelfTest[0],1); Serial.println("% of factory value");
    Serial.print("y-axis self test: acceleration trim within : ");
    Serial.print(myIMU.SelfTest[1],1); Serial.println("% of factory value");
    Serial.print("z-axis self test: acceleration trim within : ");
    Serial.print(myIMU.SelfTest[2],1); Serial.println("% of factory value");
    Serial.print("x-axis self test: gyration trim within : ");
    Serial.print(myIMU.SelfTest[3],1); Serial.println("% of factory value");
    Serial.print("y-axis self test: gyration trim within : ");
    Serial.print(myIMU.SelfTest[4],1); Serial.println("% of factory value");
    Serial.print("z-axis self test: gyration trim within : ");
    Serial.print(myIMU.SelfTest[5],1); Serial.println("% of factory value");

    // Calibrate gyro and accelerometers, load biases in bias registers
    myIMU.calibrateMPU9250(myIMU.gyroBias, myIMU.accelBias);

    Serial.print("calibrate accel bias: x: "); 
    Serial.print(myIMU.accelBias[0],1); Serial.print(", y: ");
    Serial.print(myIMU.accelBias[1],1); Serial.print(", z: ");
    Serial.print(myIMU.accelBias[2],1); Serial.println("");


    myIMU.initMPU9250();
    // Initialize device for active mode read of acclerometer, gyroscope, and
    // temperature
    Serial.println("MPU9250 initialized for active data mode....");

    // Read the WHO_AM_I register of the magnetometer, this is a good test of
    // communication
    byte d = myIMU.readByte(AK8963_ADDRESS, WHO_AM_I_AK8963);
    Serial.print("AK8963 "); Serial.print("I AM "); Serial.print(d, HEX);
    Serial.print(" I should be "); Serial.println(0x48, HEX);

  } // if (c == 0x71)
  else
  {
    Serial.print("Could not connect to MPU9250: 0x");
    Serial.println(c, HEX);
    while(1) ; // Loop forever if communication doesn't happen
  }

}

void loop() {

  // if first_round (or after a classification), buffer will be filled
  if(first_round) {
    for(; count<buff_size_start; count++) {
      myIMU.readAccelData(raw[count]); delay(INTERVAL);
    }
    count = 0;
    first_round = false;
  }
  else {
    myIMU.readAccelData(raw[count]); delay(INTERVAL);

    // classify if it is the start of a valid movement every 1 new sample
    if((count+1)%1==0) {
      uint8_t label_start = classify(raw, 1);

      // start recording data
      if(label_start==1) {
        count = buff_size_start;

        uint8_t label_end=2;
        for(; count<buff_size_move; count++) {
          if(count>=16) {
            label_end = classify(raw, 2);
            
            // stop recording data
            if(label_end==1) {
              break;
            }
          }
          
          myIMU.readAccelData(raw[count]); delay(INTERVAL);          
        }
        // adjust the number of recorded samples in the buffer
        if((count==(buff_size_move-1))&&label_end!=1) {
          count += 1;
        }

        // classify movement
        uint8_t label_move = classify(raw, 0);

        // report result of movement classification
        if(label_move==1) {
          count_press += 1;
          Serial.print("Press up: ");
          Serial.println(count_press);
        }
        if(label_move==2) {
          count_sit += 1;
          Serial.print("Sit up: ");
          Serial.println(count_sit);
        }
        if(label_move==3) {
          count_lunge += 1;
          Serial.print("Lunge: ");
          Serial.println(count_lunge);
        }

        first_round = true;
      }
    }
    count++;

    if(count>=buff_size_start) {
      count = 0;
    }
  }
}
