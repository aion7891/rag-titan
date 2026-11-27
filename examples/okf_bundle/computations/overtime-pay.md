---
type: Attested Computation
title: Overtime pay
description: How overtime pay for one month is computed.
tags: [payroll]
runtime: python
parameters:
  - { name: hours, type: number, required: true }
  - { name: hourly_rate, type: number, required: true }
attester:
  resource: /references/attesters/overtime_check.py
generated: { by: "human:payroll", at: "2026-02-01T09:00:00Z" }
verified: { by: "human:payroll-lead", at: "2026-02-02T09:00:00Z" }
---

# Computation

```python
overtime_pay = hours * hourly_rate * 1.25
```

Overtime is paid at 125% of the hourly rate.
