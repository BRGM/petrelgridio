from tests.test_export_to_vtu import test_export_to_vtu, test_dummy_grid, test_dummy_grid_fault_ramp

from tests.data.fault_pillars import get_fault_pillars_data
from geometry.petrel_mesh import mesh

if __name__ == "__main__":
    # v, e = get_fault_pillars_data()
    # mesh(v, e)
    # assert False

    # print("Running test_export_to_vtu()")
    test_export_to_vtu()

    # print("\nRunning test_dummy_grid()")
    # test_dummy_grid()

    # print("\nRunning test_dummy_grid_fault_ramp()")
    # test_dummy_grid_fault_ramp()
