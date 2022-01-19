from tests.test_export_to_vtu import test_export_to_vtu, test_dummy_grid, test_dummy_grid_fault_ramp
from tests.test_fault_pillars import test_fault_pillars

from tests.data.fault_pillars import get_fault_pillars_data
from geometry.petrel_mesh import mesh

if __name__ == "__main__":
    print("Running test_export_to_vtu()")
    test_export_to_vtu()

    # print("\nRunning test_dummy_grid()")
    # test_dummy_grid()

    # print("\nRunning test_dummy_grid_fault_ramp()")
    # test_dummy_grid_fault_ramp()

    # print("\nRunning test_fault_pillars()")
    # test_fault_pillars()
