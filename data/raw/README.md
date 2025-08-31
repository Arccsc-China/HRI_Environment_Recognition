# Dataset — raw data overview

This project uses synchronized kinematic, EMG and torque recordings.  Below is a concise description of the raw files and their array layout.

## Files

* `Kinematic/Angle.csv` — hand angle time series.
* `Kinematic/Target.csv` — target angle for the tracking task.
* `EMG/Ext.csv` — extensor muscle activation.
* `EMG/Flex.csv` — flexor muscle activation.
* `EMG/KM.csv` — muscle co-contraction: `KM(i) = min(|Ext(i)|, |Flex(i)|)`.
* `EMG/KD.csv` — reciprocal activation: `KD(i) = Ext(i) - Flex(i)`.
* `Torque/Torque_motor.csv` — motor torque produced at the handle.

## Array shape & axes

Each file is organized as an array with shape:

```
(2000, 22, 9, 9)
```

meaning:

1. **2000** — time stamps (samples)
2. **22** — subjects
3. **9** — experimental conditions, ordered as:
   `['V0H0','V0H1','V0H2','V1H0','V1H1','V1H2','V2H0','V2H1','V2H2']`
   (visual noise × haptic noise; levels `0–2`, where `0` = no noise)
4. **9** — trials per condition; **last four trials** are the stabilized trials
