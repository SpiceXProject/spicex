# Capacitor with Resistor

Philip Mocz (2026)

DC analysis of a capacitor in parallel with a resistor driven by a voltage source


## Circuit

```
 n1+----------+----------+
   |          |          |
 [ V=5V ] [ R1=1kΩ ] [ C=10µF ]
   |          |          |
 n0+----------+----------+
```


## Usage

```console
python capacitor_with_resistor.py
```


## DC Analysis

In DC steady state the capacitor is an ideal open circuit (zero current),
so the source current flows entirely through the resistor:

- V(n1) = 5 V
- I through R = 5 V / 1 kΩ = 5 mA
- I through C = 0 (capacitor blocks DC)
- I from source = 5 mA
