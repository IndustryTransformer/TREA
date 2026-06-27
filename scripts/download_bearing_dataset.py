"""Download NASA Bearing dataset - a simpler industrial sensor dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


def download_bearing_dataset(data_dir="./data/nasa_bearing"):
    """Download NASA IMS Bearing dataset.

    This dataset contains vibration sensor readings from bearings
    run to failure. It's smaller and easier to work with than turbofan.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    # IMS Bearing dataset from NASA prognostics repository
    # This is a well-known dataset for predictive maintenance

    print("NASA Bearing Dataset")
    print("=" * 60)
    print("This dataset contains vibration sensor data from bearing")
    print("run-to-failure experiments. It's commonly used for:")
    print("- Remaining Useful Life (RUL) prediction")
    print("- Anomaly detection")
    print("- Predictive maintenance")
    print("\nUnfortunately, direct download links change frequently.")
    print("\nTo get this dataset:")
    print(
        "1. Visit: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
    )
    print("2. Look for 'Bearing Data Set' or 'IMS Bearing Data'")
    print("3. Download and extract to:", data_path)

    # For now, let's create a synthetic dataset that mimics bearing data
    print("\n" + "=" * 60)
    print("Creating synthetic bearing-like dataset for demonstration...")

    create_synthetic_bearing_data(data_path)

    return data_path


def create_synthetic_bearing_data(data_path):
    """Create synthetic data that mimics bearing sensor data."""

    # Bearing dataset typically has:
    # - 4 bearings
    # - 2 accelerometers per bearing (X and Y direction)
    # - High frequency sampling (20kHz typical)
    # - Run to failure over days/weeks

    # We'll create a simplified version
    n_bearings = 4
    n_timesteps = 1000  # Simplified from millions of points

    for bearing_id in range(1, n_bearings + 1):
        bearing_data = []

        # Simulate degradation over time
        degradation_rate = np.random.uniform(0.001, 0.003)

        for t in range(n_timesteps):
            # Base vibration level increases with degradation
            base_vibration = 0.1 + degradation_rate * t

            # Add noise and periodic components
            sensor_x = base_vibration * np.random.randn() + 0.1 * np.sin(
                2 * np.pi * t / 50
            )
            sensor_y = base_vibration * np.random.randn() + 0.1 * np.cos(
                2 * np.pi * t / 50
            )

            # Add random spikes (simulating impacts)
            if np.random.random() < 0.02:
                sensor_x += np.random.uniform(1, 3)
            if np.random.random() < 0.02:
                sensor_y += np.random.uniform(1, 3)

            bearing_data.append(
                {
                    "timestamp": t,
                    "sensor_x": sensor_x,
                    "sensor_y": sensor_y,
                    "health_indicator": max(0, 1 - degradation_rate * t),
                }
            )

        # Save to CSV
        df = pd.DataFrame(bearing_data)
        df.to_csv(data_path / f"bearing_{bearing_id}.csv", index=False)

    print(f"\nCreated synthetic bearing data for {n_bearings} bearings")
    print(f"Files saved to: {data_path}")
    print("\nEach file contains:")
    print("- timestamp: Time index")
    print("- sensor_x: X-axis vibration")
    print("- sensor_y: Y-axis vibration")
    print("- health_indicator: Synthetic health score (1=healthy, 0=failed)")


if __name__ == "__main__":
    download_bearing_dataset()
