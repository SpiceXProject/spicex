# Inductor with Resistor

Philip Mocz (2026)

DC analysis of an inductor in series with a resistor driven by a voltage source.

## Circuit

```
  n1+----[ L=10mH ]----+n2
    |                  |
  [ V=5V ]           [ R=1kΩ ]
    |                  |
  n0+------------------+
```

## DC Analysis

In DC steady state the inductor is an ideal short circuit (zero voltage drop),
so the full 5 V source voltage appears across the 1 kΩ resistor:

- V(n1) = 5 V
- V(n2) = 5 V  (inductor short: V(n1) = V(n2))
- I = 5 V / 1 kΩ = 5 mA  (through inductor and resistor)


## Usage

```console
python inductor_with_resistor.py
```
