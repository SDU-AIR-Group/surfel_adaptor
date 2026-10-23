from dataclasses import asdict, dataclass
import random
from typing import Any, Sequence


@dataclass(frozen=True)
class LidarProfile:
    manufacturer: str
    model: str
    channels: int
    horizontal_fov_deg: float
    horizontal_resolution_deg: float
    vertical_fov_deg: float
    vertical_resolution_deg: float
    rotations_per_second: float
    distance_lower_m: float
    distance_upper_m: float
    range_noise_std_m: float

    def scan_kwargs(self) -> dict[str, float]:
        """Return the profile fields consumed by ``scan_rotating``."""
        return {
            "xStepDegree": self.horizontal_resolution_deg,
            "fovX": self.horizontal_fov_deg,
            "yStepDegree": self.vertical_resolution_deg,
            "fovY": self.vertical_fov_deg,
            "rotationsPerSecond": self.rotations_per_second,
            "distanceLower": self.distance_lower_m,
            "distanceUpper": self.distance_upper_m,
            "sigma": self.range_noise_std_m,
        }

    def metadata(self) -> dict[str, Any]:
        return asdict(self)


# Values model public nominal operating modes.  range_noise_std_m is a
# Gaussian approximation of each product family's published ranging
# accuracy/precision rather than a manufacturer-provided noise distribution.
LIDAR_PROFILES: tuple[LidarProfile, ...] = (
    LidarProfile("Ouster", "OS0-32", 32, 360.0, 0.09, 90.0, 90.0 / 31, 10.0, 0.8, 100.0, 0.05),
    LidarProfile("Ouster", "OS0-64", 64, 360.0, 0.09, 90.0, 90.0 / 63, 10.0, 0.8, 100.0, 0.05),
    LidarProfile("Velodyne", "VLP-16", 16, 360.0, 0.20, 30.0, 30.0 / 15, 10.0, 1.0, 100.0, 0.03),
    LidarProfile("Velodyne", "VLP-32C", 32, 360.0, 0.20, 40.0, 40.0 / 31, 10.0, 0.8, 200.0, 0.03),
    LidarProfile("Velodyne", "HDL-64E", 64, 360.0, 0.09, 26.9, 26.9 / 63, 10.0, 1.0, 120.0, 0.02),
    LidarProfile("Hesai", "PandarXT-16", 16, 360.0, 0.18, 30.0, 30.0 / 15, 10.0, 0.05, 100.0, 0.02),
    LidarProfile("Hesai", "PandarXT-32", 32, 360.0, 0.18, 31.0, 31.0 / 31, 10.0, 0.05, 120.0, 0.01),
    LidarProfile("Hesai", "Pandar64", 64, 360.0, 0.20, 40.0, 40.0 / 63, 10.0, 0.3, 200.0, 0.02),
    LidarProfile("RoboSense", "Helios-16", 16, 360.0, 0.20, 30.0, 30.0 / 15, 10.0, 0.2, 120.0, 0.03),
    LidarProfile("RoboSense", "Helios-32", 32, 360.0, 0.20, 26.0, 26.0 / 31, 10.0, 0.2, 120.0, 0.03),
    LidarProfile("RoboSense", "RS-LiDAR-64", 64, 360.0, 0.20, 26.8, 26.8 / 63, 10.0, 0.4, 200.0, 0.03),
)


def choose_lidar(rng: Any = random) -> LidarProfile:
    """Select one profile uniformly; ``rng`` makes deterministic tests simple."""
    return rng.choice(LIDAR_PROFILES)


def validate_profiles(profiles: Sequence[LidarProfile] = LIDAR_PROFILES) -> None:
    names = set()
    for profile in profiles:
        if profile.model in names or not 16 <= profile.channels <= 64:
            raise ValueError(f"Invalid LiDAR profile: {profile.model}")
        names.add(profile.model)
        if any(value <= 0 for value in profile.scan_kwargs().values()):
            raise ValueError(f"Non-positive scan value: {profile.model}")
