# RC Step Response

Philip Mocz (2026)

Transient simulation of an RC circuit driven by a 1 V step voltage source


## Usage

```console
python rc_step_response.py
```


## Circuit

```
  n1+------[ R=1kΩ ]------+n2
    |                     |
  [ V=1V ]              [ C=1µF ]
    |                     |
  n0+---------------------+
```


## Transient Analysis

The capacitor charges through the resistor with time constant τ = RC = 1 ms.

Analytic solution: V_C(t) = 1 − exp(−t / τ)
