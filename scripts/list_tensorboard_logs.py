"""List all available TensorBoard logs."""

import os

from pathlib import Path


def list_tensorboard_logs(log_dir="logs"):
    """List all TensorBoard experiment logs."""
    log_path = Path(log_dir)

    if not log_path.exists():
        print(f"Log directory '{log_dir}' does not exist.")
        return

    print("Available TensorBoard logs:")
    print("=" * 60)

    experiments = {}

    # Walk through the log directory
    for experiment_dir in sorted(log_path.iterdir()):
        if experiment_dir.is_dir():
            versions = []
            for version_dir in sorted(experiment_dir.iterdir()):
                if version_dir.is_dir() and version_dir.name.startswith("version_"):
                    # Check if it has event files
                    event_files = list(version_dir.glob("events.out.tfevents.*"))
                    if event_files:
                        # Get the most recent event file
                        latest_event = max(event_files, key=os.path.getmtime)
                        mod_time = Path(latest_event).stat().st_mtime
                        versions.append((version_dir.name, mod_time))

            if versions:
                experiments[experiment_dir.name] = versions

    # Display the experiments
    for exp_name, versions in experiments.items():
        print(f"\n📊 {exp_name}:")
        for version, mod_time in sorted(versions, key=lambda x: x[1], reverse=True):
            from datetime import datetime

            dt = datetime.fromtimestamp(mod_time)
            print(f"   - {version} (last modified: {dt.strftime('%Y-%m-%d %H:%M:%S')})")

    print("\n" + "=" * 60)
    print("\nTo view a specific experiment, run:")
    print("  tensorboard --logdir logs/<experiment_name>")
    print("\nTo view all experiments, run:")
    print("  tensorboard --logdir logs")


if __name__ == "__main__":
    list_tensorboard_logs()
