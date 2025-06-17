def run():
    print("Generating bootstrap data...")
    import numpy as np
    # Step 1: Resample country-year pairs with replacement
    # Step 2: Recompute model coefficients on each resample
    # Step 3: Save bootstrap estimates to disk
