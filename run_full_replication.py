from data_processing import generate_figure2_data, generate_bootstrap_data, get_temperature_change
from projections import compute_main_projections, compute_damage_function
from figures import make_figure2, make_figure3, make_figure4, make_figure5

def run_full_replication():
    print("Step 1: Generating regression input data (Figure 2)...")
    generate_figure2_data.run()

    print("Step 2: Generating bootstrap datasets...")
    generate_bootstrap_data.run()

    print("Step 3: Loading IAM temperature change scenarios...")
    get_temperature_change.run()

    print("Step 4: Computing main projections under warming scenarios...")
    compute_main_projections.run()

    print("Step 5: Computing damage function from projections...")
    compute_damage_function.run()

    print("Step 6: Generating figures...")
    make_figure2.run()
    make_figure3.run()
    make_figure4.run()
    make_figure5.run()

if __name__ == "__main__":
    run_full_replication()
