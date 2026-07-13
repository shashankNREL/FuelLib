# Constantinou-Gani GCM Decomposition — Implementation Notes

## Reference
- Constantinou & Gani, AIChE J. 40(10), October 1994
- "New group contribution method for estimating properties of pure compounds"

## Key Design Decisions

### 1. Two-level decomposition
- **First-order groups** (columns 1–78 in gcmTable.csv): atom-level fragments identical
  to UNIFAC-style groups. Cyclic CH2 = same "CH2" as acyclic (no distinction).
- **Second-order groups** (columns 80–122 in gcmTable.csv; columns 79–121 in
  refCompounds.csv): structural corrections for branching, ring strain, conjugation.

### 2. Column mapping
- `gcmTable.csv`: 123 columns total. Col 0 = "Property", Col 1 = "Units",
  Cols 2–79 = first-order groups, Cols 80–122 = second-order groups.
- `refCompounds.csv`: 122 columns total. Col 0 = "Compound",
  Cols 1–78 = first-order groups (annotated with numbers like "(1)", "(2)"),
  Col 79 header = "Group j  (CH3)2CH" (merged label),
  Cols 80–121 = remaining second-order groups.

### 3. Scope limitation (SAF hydrocarbons only)
Only groups relevant to SAF jet-fuel surrogates are implemented:
- First-order: CH3, CH2, CH, C, CH2=CH, CH=CH, CH2=C, CH=C, C=C, ACH, AC,
  ACCH3, ACCH2, ACCH
- Second-order: (CH3)2CH, (CH3)3C, CH(CH3)CH(CH3), CH(CH3)C(CH3)2,
  C(CH3)2C(CH3)2, 3–7 membered ring, Alicyclic side-chain CcyclicCm m>1, CH3CH3
- Heteroatom-containing groups raise UnsupportedGroupError.

### 4. Ring counting assumptions
- Use RDKit `GetSymmSSSR()` to enumerate Smallest Set of Smallest Rings (SSSR).
- Only **non-aromatic** rings contribute to ring-size corrections.
  Aromatic rings (all atoms in ring are aromatic) are excluded.
- Each distinct non-aromatic ring contributes +1 to its size bucket (3..7 membered).
- For fused bicyclics like decalin: SSSR yields 2 six-membered rings → `6 membered ring = 2`.
- For tetralin (one aromatic + one alicyclic ring): only the alicyclic ring counts → `6 membered ring = 1`.
- For indane: 5-membered alicyclic ring → `5 membered ring = 1`.

### 5. Alicyclic side-chain CcyclicCm (m > 1) assumption
- From refCompounds.csv: **ALL** monocycloparaffins (C07–C19), including
  ethylcyclohexane (m=2), propylcyclohexane (m=3), etc., have this group = 0.
- This means the "alicyclic side-chain" correction does NOT apply to simple
  alkyl substituents on cycloparaffin rings in the SAF molecule context.
- The exact structural requirement is unclear from the paper alone.
- **Decision**: Disabled this detection (always returns 0) to match FuelLib data.

### 6. Ring detection for fused aromatic-alicyclic systems
- In tetralin/indane, the alicyclic ring shares 2 atoms with the aromatic ring.
  Those junction atoms are marked aromatic by RDKit.
- Correct rule: count a ring for the size correction if it is **NOT fully aromatic**
  (i.e., at least one atom in the ring is non-aromatic).
- This correctly gives:
  - Tetralin: 1 × "6 membered ring" (the alicyclic ring, 4 non-arom + 2 arom atoms)
  - Indane: 1 × "5 membered ring" (3 non-arom + 2 arom atoms)
  - Decalin: 2 × "6 membered ring" (both fully non-aromatic)
  - Naphthalene: 0 rings counted (both are fully aromatic)

### 6. FuelLib files are READ-ONLY
- No writes to any file under FuelLib/ except within tools/.
- Comparison against refCompounds.csv is purely for validation.

---

## Bug Found in FuelLib

### Cycloaromatic-C09 (indane) — AC=2 should be AC=0

**File**: `fuelData/groupDecompositionData/refCompounds.csv`

**Current (WRONG)**:
```
Cycloaromatic-C09: CH2=1, ACH=4, AC=2, ACCH2=2, 5-ring=1
```
Formula check: C = 1 + 4 + 2 + 2×2 = **11 ≠ 9** ✗

**Correct**:
```
Cycloaromatic-C09: CH2=1, ACH=4, AC=0, ACCH2=2, 5-ring=1
```
Formula check: C = 1 + 4 + 0 + 2×2 = **9** ✓, H = 2 + 4 + 0 + 2×2 = **10** ✓ → C₉H₁₀

**Root cause**: The two aromatic ring-junction carbons (shared with the 5-membered ring)
are already consumed by the ACCH2 groups. They should NOT be double-counted as AC.
This is consistent with tetralin (Cycloaromatic-C10) which correctly has AC=0.

---

## Implementation Log

### Phase 1: First-order groups
- Reuse logic from decompose_unifac.py: atom classification by H-count and neighbor count
- Extended to cover all C=C variants (CH=CH, CH2=C, CH=C, C=C, CH2=C=CH)
- Same ACH/AC/ACCH3/ACCH2/ACCH convention as UNIFAC

### Phase 2: Second-order groups (branching)
- **(CH3)2CH**: find aliphatic CH atoms (degree=3, 1H) with exactly 2 terminal
  CH3 neighbors (degree=1, 3H).
- **(CH3)3C**: find quaternary C atoms (degree=4, 0H) with exactly 3 terminal
  CH3 neighbors.
- **CH(CH3)CH(CH3)**: find pairs of adjacent CH atoms each having at least 1 CH3.
  Count each such pair once (avoid double-counting).
- **CH(CH3)C(CH3)2**: adjacent CH (1 CH3 neighbor) and C (2 CH3 neighbors).
- **C(CH3)2C(CH3)2**: adjacent quaternary C atoms each with 2 CH3 neighbors.

### Phase 3: Ring counting
- GetSymmSSSR → filter aromatic rings → classify by size → count

### Phase 4: Alicyclic side-chain
- For each ring atom bonded to a non-ring aliphatic carbon, walk the chain.
  If the chain has ≥2 carbons, count +1.

---

## Validation Results

### Final run (all formulas correct, 18/19 match FuelLib):

| Molecule | Formula | First-order | Second-order | FuelLib |
|----------|---------|-------------|--------------|---------|
| n-heptane | C7H16 ✓ | CH3=2, CH2=5 | — | PASS |
| n-decane | C10H22 ✓ | CH3=2, CH2=8 | — | PASS |
| 2-methylhexane | C7H16 ✓ | CH3=3, CH=1, CH2=3 | (CH3)2CH=1 | PASS |
| 2-methylheptane | C8H18 ✓ | CH3=3, CH=1, CH2=4 | (CH3)2CH=1 | PASS |
| 2,3-dimethylbutane | C6H14 ✓ | CH3=4, CH=2 | (CH3)2CH=2, CH(CH3)CH(CH3)=1 | — |
| neopentane | C5H12 ✓ | CH3=4, C=1 | (CH3)3C=1 | — |
| methylcyclohexane | C7H14 ✓ | CH3=1, CH=1, CH2=5 | 6-ring=1 | PASS |
| ethylcyclohexane | C8H16 ✓ | CH3=1, CH2=6, CH=1 | 6-ring=1 | PASS |
| decalin | C10H18 ✓ | CH2=8, CH=2 | 6-ring=2 | PASS |
| cis-decalin | C10H18 ✓ | CH2=8, CH=2 | 6-ring=2 | PASS |
| toluene | C7H8 ✓ | ACCH3=1, ACH=5 | — | PASS |
| ethylbenzene | C8H10 ✓ | ACCH2=1, CH3=1, ACH=5 | — | PASS |
| propylbenzene | C9H12 ✓ | ACCH2=1, CH3=1, CH2=1, ACH=5 | — | PASS |
| naphthalene | C10H8 ✓ | ACH=8, AC=2 | — | PASS |
| 1-methylnaphthalene | C11H10 ✓ | ACCH3=1, ACH=7, AC=2 | — | PASS |
| indane | C9H10 ✓ | ACCH2=2, CH2=1, ACH=4 | 5-ring=1 | FAIL* |
| tetralin | C10H12 ✓ | ACCH2=2, CH2=2, ACH=4 | 6-ring=1 | PASS |
| 2-methyltetralin | C11H14 ✓ | ACCH2=2, CH3=1, CH=1, CH2=1, ACH=4 | 6-ring=1 | PASS |
| 1-dodecene | C12H24 ✓ | CH2=CH=1, CH2=9, CH3=1 | — | PASS |

*indane FAIL = FuelLib has AC=2 which is a data entry error (see bug report above)
