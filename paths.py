import os
import sys

FUELLIB_DIR = os.path.dirname(__file__)
GCMTABLE_DIR = os.path.join(FUELLIB_DIR, "gcmTableData")
UNIFAC_TABLE_DIR = os.path.join(FUELLIB_DIR, "unifacTableData")
UNIFAC_SUBGROUP_FILE = os.path.join(UNIFAC_TABLE_DIR, "unifac_subgroups.csv")
UNIFAC_AMN_FILE = os.path.join(UNIFAC_TABLE_DIR, "unifac_amn.csv")
SOURCE_DIR = os.path.join(FUELLIB_DIR, "source")
FUELDATA_DIR = os.path.join(FUELLIB_DIR, "fuelData")
FUELDATA_GC_DIR = os.path.join(FUELDATA_DIR, "gcData")
FUELDATA_DECOMP_DIR = os.path.join(FUELDATA_DIR, "groupDecompositionData")
FUELDATA_UNIFAC_DIR = os.path.join(FUELDATA_DIR, "unifacDecomposition")
FUELDATA_PROPS_DIR = os.path.join(FUELDATA_DIR, "propertiesData")
TESTS_DIR = os.path.join(FUELLIB_DIR, "tests")
TESTS_BASELINE_DIR = os.path.join(TESTS_DIR, "baselinePredictions")
TUTORIALS_DIR = os.path.join(FUELLIB_DIR, "tutorials")

sys.path.append(SOURCE_DIR)
