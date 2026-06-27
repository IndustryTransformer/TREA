"""Natural-language descriptions for 3W sensor column codes.

The 3W dataset uses cryptic Portuguese-derived sensor codes (``P-PDG``,
``ABER-CKGL``, ``ESTADO-DHSV``). Generic text encoders extract almost no
signal from the raw codes, so semantic column embeddings only become
meaningful when the codes are expanded into descriptions that name the
physical quantity (pressure / temperature / flow / valve-state /
choke-opening) and its location in the well.

Naming convention (for reference):
    P-*      pressure            T-*      temperature
    Q*       flow rate           ABER-*   choke opening ("abertura")
    ESTADO-* equipment state     PT-*     pressure transmitter
    JUS      downstream ("jusante")       MON   upstream ("montante")
    CKP      production choke    CKGL     gas-lift choke
    PDG      permanent downhole gauge     TPT   downhole P/T transducer
    DHSV     downhole safety valve        SDV   shutdown valve
    GL       gas lift            BS       subsea booster pump
    M1/M2    master valves       W1/W2    wing valves   XO/PXO  crossover
"""

from __future__ import annotations


# Maps each 3W sensor code to a natural-language description. The descriptions
# emphasise the semantic axis that transfers across schemas: what is measured
# (pressure/temperature/flow/state) and where (upstream/downstream, downhole,
# annulus, choke, valve).
SENSOR_DESCRIPTIONS: dict[str, str] = {
    "ABER-CKGL": "opening of the gas-lift injection choke valve",
    "ABER-CKP": "opening of the production choke valve",
    "ESTADO-DHSV": "state of the downhole safety valve",
    "ESTADO-M1": "state of the production master valve",
    "ESTADO-M2": "state of the annulus master valve",
    "ESTADO-PXO": "state of the pig crossover valve",
    "ESTADO-SDV-GL": "state of the gas-lift shutdown valve",
    "ESTADO-SDV-P": "state of the production shutdown valve",
    "ESTADO-W1": "state of the production wing valve",
    "ESTADO-W2": "state of the annulus wing valve",
    "ESTADO-XO": "state of the crossover valve",
    "P-ANULAR": "pressure in the well annulus",
    "P-JUS-BS": "pressure downstream of the subsea booster pump",
    "P-JUS-CKGL": "pressure downstream of the gas-lift choke",
    "P-JUS-CKP": "pressure downstream of the production choke",
    "P-MON-CKGL": "pressure upstream of the gas-lift choke",
    "P-MON-CKP": "pressure upstream of the production choke",
    "P-MON-SDV-P": "pressure upstream of the production shutdown valve",
    "P-PDG": "pressure at the permanent downhole gauge",
    "PT-P": "pressure at the production line transmitter",
    "P-TPT": "pressure at the downhole temperature and pressure transducer",
    "QBS": "flow rate through the subsea booster pump",
    "QGL": "gas-lift injection flow rate",
    "T-JUS-CKP": "temperature downstream of the production choke",
    "T-MON-CKP": "temperature upstream of the production choke",
    "T-PDG": "temperature at the permanent downhole gauge",
    "T-TPT": "temperature at the downhole temperature and pressure transducer",
}


def describe_columns(column_names: list[str]) -> list[str]:
    """Return descriptions aligned to ``column_names``.

    Unknown codes fall back to a normalised form of the code itself so the
    encoder still receives readable text instead of crashing.

    Args:
        column_names: Ordered sensor codes (must match the data feature axis).

    Returns:
        List of natural-language descriptions, one per column.
    """
    descriptions = []
    for name in column_names:
        if name in SENSOR_DESCRIPTIONS:
            descriptions.append(SENSOR_DESCRIPTIONS[name])
        else:
            # Fallback: turn "P-MON-CKP" into "p mon ckp" so the encoder gets
            # something tokenizable rather than an unknown symbol.
            descriptions.append(name.lower().replace("-", " ").replace("_", " "))
    return descriptions
