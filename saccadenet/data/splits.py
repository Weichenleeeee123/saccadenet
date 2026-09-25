"""Fixed MNIST source-index partitions from the experiment protocol."""


def split_digit_ids() -> dict[str, range]:
    return {
        "train": range(0, 50_000),
        "calibration": range(50_000, 55_000),
        "development": range(55_000, 60_000),
        "final_test": range(0, 10_000),
    }
