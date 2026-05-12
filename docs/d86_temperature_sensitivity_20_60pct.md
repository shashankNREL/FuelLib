# D86 Temperature Prediction Sensitivity: 20–60% Volume Distilled

**Fuel:** POSF-10289  
**Model:** RK2 + lumped condenser + Option E controller  
**Calibrated parameters:** C_glass=0.548, h_multiplier=1.9404, use_srk=True, Q1=76.54 W, UA_cond=8.186 W/K, T_bath=281.723 K, T_bubble_hi=658.12 K, dt=1.0 s, T_room=298.15 K

This document identifies the physics components and parameters most likely to improve T_D86 predictions in the 20–60% volume-distilled range, where comparisons with NJFCP experimental data remain imperfect.

---

## Physical context: what is happening at 20–60%

At 20% collected the pot has been depleted of light aromatics and light paraffins (C7–C9). The dominant volatile species in this window are **C10–C13 monocycloparaffins** (~30% by weight total), **C10–C14 isoparaffins**, **C10–C13 n-paraffins**, and **C10–C13 dicycloparaffins** (~15%). The bubble-point temperature is rising from roughly 170 °C toward 220 °C. This is the steepest-composition-change part of the curve where every model stage contributes.

---

## Options for improvement, in order of likely impact

### 1. psat accuracy for mid-weight cycloparaffins (Stage 1 — highest leverage)

`psat()` uses the Lee-Kesler correlation with GCM-estimated `Tc`, `Pc`, `omega`. For cycloparaffins (the dominant species in this window), the Constantinou–Gani GCM tends to underestimate `Tc` and `omega` for ring structures, which overestimates psat and therefore underestimates the bubble-point temperature T1. A systematic 3–5% error in psat for C10–C13 cycloparaffins shifts T_D86 by 4–8 °C throughout this range.

Options to explore:
- Switch to the Ambrose–Walton correlation (`psat(..., correlation="Ambrose-Walton")`) and compare with experiment.
- Replace GCM-predicted `Tc`/`Pc`/`omega` with DIPPR literature values for the key cycloparaffin pseudo-components (e.g., pentyl cyclohexane, hexyl cyclohexane, decalin, 2-methyldecalin).

### 2. UNIFAC activity coefficients (Stage 1 bubble point)

At 20–60%, the pot liquid contains aromatics (~10% remaining), cycloparaffins, and paraffins — three chemical families with non-trivial cross-interactions. The UNIFAC model in `activity()` uses a group-interaction matrix `unifac_a`. For aromatic-ring / cycloalkyl-ring / n-alkyl combinations at 450–480 K, activity coefficients can deviate from unity by 5–15%, which directly shifts the bubble point.

Options to explore:
- Verify that the `unifac_a` matrix contains entries for the CH2 (cyclic), ACH, and ACCH2 groups that appear in monocycloparaffin and aromatic pseudo-components.
- Compare the UNIFAC-predicted activity coefficients for key binary pairs (e.g., hexyl cyclohexane + n-dodecane) against available experimental data.
- Toggle `use_srk=True` to cross-check whether the SRK fugacity approach gives a more consistent bubble point for this composition range (SRK handles non-ideality via fugacity coefficients rather than activity coefficients).

### 3. h_multiplier — column neck reflux fraction (Stage 2)

`h_multiplier` scales the Churchill–Chu natural-convection coefficient for the exposed column neck. More heat loss produces more reflux R2, which returns lighter components to the pot, slowing the compositional shift and depressing T_D86 in the 20–60% window. At the neck temperatures typical of this range (T2 ~ 450–490 K), natural convection is most active.

- If predictions are **too low** in this range: reduce `h_multiplier` → less reflux → faster temperature rise.
- If predictions are **too high**: increase `h_multiplier`.
- The current calibrated value (1.94) is already substantially above unity; small changes (±0.1) have measurable effects.

### 4. C_glass — thermocouple thermal lag (Stage 3)

`C_glass` sets the time constant of the first-order CSTR that models the thermometer bulb:

```
τ = C_glass / (D2 · Cp_v)
```

With D2 ~ 2×10⁻⁴ mol/s and Cp_v ~ 350 J/mol/K, τ ≈ 8 s at current settings. In the 20–60% range T_D86 is rising (positive dT/dV), so a larger C_glass causes T_D86 to lag behind the true vapor temperature T2 — predictions appear systematically low.

- If the simulated curve is consistently too cold relative to experiment: decrease `C_glass`.
- If it converges too quickly and overshoots at 20%: increase `C_glass`.
- The physically meaningful range for a glass thermometer bulb is roughly 0.3–0.8 J/K.

### 5. Vapor Cp model (Stage 2 energy balance, Stage 3, condenser NTU)

`calculate_vapor_heat_capacity()` falls back to the **liquid-Cp proxy** when no NASA7 mechanism YAML is provided. This proxy underestimates ideal-gas Cp by approximately 15% at 450–480 K. The underestimate propagates in two places:

- **Stage 3:** lower Cp_v → smaller C_flow = D2·Cp_v → thermocouple responds more slowly → T_D86 is depressed.
- **Condenser (Stage 2b):** lower Cp_v → lower C_min → higher NTU → over-condensation → less forward vapor.

If the NASA7 mechanism YAML is not wired up (check whether `NASA7 vapor Cp enabled` was printed at fuel load time), enabling it is a free accuracy improvement. Set `f_obj.mechanism_yaml = <path_to_mechanism.yaml>` before running.

### 6. Latent heat via Watson correlation (Stage 1 energy balance)

The Watson scaling:

```
Lv_i(T) = Lv_stp · ((1 − Tr) / (1 − Trb))^0.38
```

uses a fixed exponent of 0.38. For cycloparaffins the literature-reported exponent spans 0.36–0.41, and the GCM-predicted `Lv_stp` for ring structures can carry ±5% error. An overestimated Lv makes the Option E feed-forward controller pump in more Q1 than physically needed, raising the vapor rate and compressing the temperature-volume profile in the 20–60% window.

Options to explore:
- Use DIPPR latent-heat values at the normal boiling point for the dominant cycloparaffin pseudo-components.
- Test sensitivity of exponent (0.36 vs 0.38 vs 0.40) for the ring-compound subsets.

### 7. UA_cond and T_bath (Stage 2b condenser)

`UA_cond` and `T_bath` primarily control whether the condenser outlet temperature is low enough to fully condense the incoming vapor. In the 20–60% range, most species are below their dew points at condenser conditions so V_frac ≈ 0 and the direct effect on T_D86 is second-order (T_D86 is governed by T2, upstream of the condenser).

However, these parameters directly set the **distillate volume accumulation rate**. If insufficient condensation occurs, volume is undercounted and the temperature-vs-percent curve appears stretched. If predictions are accurate in temperature but shifted along the volume axis, adjust UA_cond or T_bath first.

### 8. Target distillation rate (Option E controller)

`target_rate_ml_min=4.5 mL/min` sets the feed-forward target. If the actual experimental rate in the 20–60% window was faster (e.g., 5 mL/min during the initial steady-state settling), the controller imposes an artificially slow rate that distorts how quickly the temperature-volume profile evolves. Comparing the `rate_ml_min` history output from the simulation against any recorded experimental rate in this window would identify this source of error.

### 9. GCM critical property errors for dicycloparaffins

The C10–C14 dicycloparaffins (decalin, 2-methyldecalin, 2-ethyldecalin at ~15% combined weight) are particularly challenging for GCM. Their rigid bicyclic structure means Constantinou–Gani predictions of `Tc` and `Pc` carry larger errors than for monocyclic or linear species, propagating into both psat and latent heat. Using DIPPR values for at least decalin and 2-methyldecalin as anchors for the dicycloparaffin pseudo-component group would test whether this sub-family is the source of systematic offset.

---

## Summary table

| Stage | Component | Parameter | Direction to raise T_D86 in 20–60% | Primary uncertainty source |
|---|---|---|---|---|
| 1 — Bubble point | Lee-Kesler psat | Tc, omega (GCM cycloparaffins) | Increase Tc → lower psat → higher T_bub | GCM ring-structure error |
| 1 — Bubble point | UNIFAC activity | Interaction matrix γ | γ < 1 raises T_bub | Group-interaction accuracy |
| 2 — Neck reflux | Natural convection | h_multiplier | Decrease → less reflux → faster T rise | Geometry and flow regime |
| 3 — Thermocouple | CSTR lag | C_glass | Decrease → faster tracking → higher T_D86 | Bulb thermal mass unknown |
| 3 + condenser | Vapor heat capacity | Cp_v (NASA7 vs liquid proxy) | Enable NASA7 → increases Cp_v → faster tracking | Missing mechanism YAML |
| 2b — Condenser | Volume accumulation | UA_cond, T_bath | Increase UA_cond → more condensate → faster V | Calibrated; second-order on T |
| 1 — Energy balance | Watson Lv | Lv_stp (GCM), exponent | Correct Lv_stp → fixes Q1_ff accuracy | GCM ring-structure error |
| All | VLE method | use_srk | Compare SRK vs Raoult+UNIFAC | EoS selection |

The **psat accuracy for C10–C13 cycloparaffins** (via Tc and omega) and **C_glass** are the two parameters most likely to explain a systematic offset in the 20–60% range, because they control the floor temperature (T1) and the thermocouple tracking lag respectively. Enabling the NASA7 vapor Cp (if not already active) is a low-cost fix that improves both the condenser NTU calculation and the Stage 3 response time.
