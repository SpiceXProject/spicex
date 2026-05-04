# Resistor Sweep

Philip Mocz (2026)

Do a parallel sweep through the circuit solver to find the load resistance R_L
that maximizes power delivered to it from a voltage source


## Circuit

```
 n1+----[ R_s=1kΩ ]----+n2
   |                   |
 [ V=10V ]        [ R_L (optimized) ]
   |                   |
 n0+-------------------+
```


## Usage

```console
python resistor_sweep.py
```


## Results

Analytic result: max power transfer when `R_L = R_s`

Maximum power: `P_max = V_s^2 / (4 * R_s) = 25 mW`

c.f. `examples/maxium_power_transfer/`


## Reference

https://en.wikipedia.org/wiki/Maximum_power_transfer_theorem
