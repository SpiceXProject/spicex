# RLC Series Circuit

Philip Mocz (2026)

Transient simulation of a series RLC circuit driven by a 1 V step voltage source.


## Circuit

```
  +n1---[ L=10mH ]---+n2---[ R=2Ω ]---+n3
  |                                   |
[ V=1V ]                            [ C=100µF ]
  |                                   |
  n0+---------------------------------+
```

## Parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| ω₀ | 1000 rad/s | Natural frequency 1/√(LC) |
| ζ | 0.1 | Damping ratio R/(2√(L/C)) |
| τ | 10 ms | Envelope time constant 1/(ζω₀) |
| T₀ | 6.28 ms | Oscillation period 2π/ω₀ |

## Transient Analysis

With ζ = 0.1 < 1 the circuit is **underdamped**: the capacitor voltage oscillates
before settling to V_S = 1 V.

Analytic solution:

```
V_C(t) = V_S [1 − e^(−αt)(cos ω_d t + (α/ω_d) sin ω_d t)]
```

where α = ζω₀ = 100 rad/s and ω_d = ω₀√(1−ζ²) ≈ 995 rad/s.

The first overshoot peak occurs at t ≈ π/ω_d ≈ 3.16 ms:

```
V_C_peak = V_S (1 + e^(−απ/ω_d)) ≈ 1.73 V
```

## Result

![rlc_series_circuit](rlc_series_circuit.png)
