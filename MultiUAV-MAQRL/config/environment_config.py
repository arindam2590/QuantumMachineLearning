"""Central configuration for town geometry, drones, and episodes."""
from dataclasses import dataclass, field, fields
import math


def positive(name, value, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or (integer and not isinstance(value, int)):
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'}")



@dataclass
class TownConfig:
    building_density: float = 0.55
    blocks_per_side: int = 15
    plot_spacing: float = 13.0
    road_width: float = 4.0
    min_building_height: float = 5.0
    max_building_height: float = 35.0
    seed: int = 7

    def validate(self):
        positive("blocks_per_side", self.blocks_per_side, integer=True)
        for name in ("plot_spacing", "road_width", "min_building_height", "max_building_height"):
            positive(name, getattr(self, name))
        if not math.isfinite(self.building_density) or not 0.0 <= self.building_density <= 1.0:
            raise ValueError('building_density must lie in [0, 1]')
        if self.blocks_per_side < 1 or self.plot_spacing <= self.road_width + 4.0:
            raise ValueError('Invalid town plot count or spacing/road width')
        if not 0 < self.min_building_height < self.max_building_height:
            raise ValueError('Building height bounds must be positive and increasing')


@dataclass
class DroneConfig:
    max_roll: float = .35  # rad, desired orientation
    max_pitch: float = .35  # rad, desired orientation
    max_angular_rate: float = 3.0  # rad/s, body roll/pitch reference
    max_collective_thrust: float | None = None  # N; None uses mass*g*max_thrust_factor
    max_body_torques: tuple[float, float, float] | None = None  # Nm; None uses max_torque
    max_jerk: float | None = None  # m/s³, Euclidean command slew limit
    velocity_mode: str = "pd"
    velocity_integral_error_gate: float = .3  # m/s, integrate only near tracking
    velocity_ki: float = .4
    velocity_integral_limit: float = 1.0  # m, integrated velocity error per axis
    velocity_kp: float = 1.5
    velocity_kd: float = 0.05
    minimum_specific_thrust: float = 0.1
    hover_altitude_kp: float = 2.0
    hover_vertical_kd: float = 2.5
    yaw_target: float = 0.0
    min_thrust_projection: float = 0.2
    gravity: float = 9.81

    @property
    def inertia_diagonal(self):
        """Uniform solid-sphere inertia matching the physical collision sphere."""
        return (2*self.mass*self.radius**2/5,)*3

    min_altitude: float = 0.7
    max_altitude: float = 85.0
    response_time: float | None = None  # None retains legacy 1 / vel_gain.
    mass: float = 1.2
    radius: float = 0.55
    visual_scale: float = 4.0  # Visual appearance only; collision geometry is unchanged.
    max_speed_xy: float = 5.0
    max_speed_z: float = 2.5
    max_yaw_rate: float = 1.2
    max_tilt: float = 0.35
    max_accel_xy: float = 3.0
    max_accel_z: float = 4.0
    vel_gain: float = 1.5
    attitude_kp: float = 9.0
    attitude_kd: float = 3.5
    yaw_rate_gain: float = 1.8
    max_torque: float = 3.0
    max_thrust_factor: float = 2.2


    def validate(self):
        for item in fields(self):
            if item.name == 'velocity_mode':
                if self.velocity_mode not in ('pd', 'pid'):
                    raise ValueError('velocity_mode must be pd or pid')
                continue
            if item.name in ('max_collective_thrust', 'max_jerk', 'max_body_torques'):
                value = getattr(self, item.name)
                if value is None:
                    continue
                if item.name == 'max_body_torques':
                    if len(value) != 3:
                        raise ValueError('max_body_torques needs three Nm limits')
                    for limit in value:
                        positive(item.name, limit)
                else:
                    positive(item.name, value)
                continue
            if item.name == 'velocity_kd' and self.velocity_kd == 0:
                continue
            if item.name == 'yaw_target':
                if not math.isfinite(self.yaw_target):
                    raise ValueError('yaw_target must be finite')
                continue
            if item.name == "response_time" and self.response_time is None:
                continue
            positive(item.name, getattr(self, item.name))
        if any(not 0 < angle < math.pi/2 for angle in (self.max_tilt, self.max_roll, self.max_pitch)):
            raise ValueError('max_tilt must be between zero and pi/2')
        if self.min_thrust_projection > 1 or self.max_thrust_factor < 1:
            raise ValueError('Invalid thrust projection or maximum hover thrust factor')
        if self.min_altitude < self.radius or self.max_altitude <= self.min_altitude:
            raise ValueError("Altitude bounds must clear ground and be increasing")


@dataclass
class RegionConfig:
    """Box center and half dimensions in metres; None uses boundary-road defaults."""
    grid_shape: tuple[int, int] = (2, 3)  # Depth (x), width (y), in town plots.
    center: tuple[float, float, float] | None = None
    half_size: tuple[float, float, float] | None = None

    def validate(self):
        if (len(self.grid_shape) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1
                                           for x in self.grid_shape)):
            raise ValueError("grid_shape must contain two positive plot counts")
        for name in ("center", "half_size"):
            value = getattr(self, name)
            if value is not None:
                if len(value) != 3 or not all(math.isfinite(x) for x in value):
                    raise ValueError(f"{name} must contain three finite coordinates")
                if name == "half_size" and any(x <= 0 for x in value):
                    raise ValueError("half_size must be positive")


@dataclass
class SensorConfig:
    """Reserved settings; sensors are not activated in this configuration step."""
    enabled: bool = False
    lidar_enabled: bool = True
    lidar_hz: float = 10.
    lidar_rays: int = 36
    lidar_fov: float = 360.
    lidar_vertical_angles: tuple[float, ...] = (-20., 0., 20.)
    lidar_range: float = 30.
    lidar_noise_std: float = 0.
    camera_enabled: bool = False
    camera_hz: float = 5.
    camera_width: int = 64
    camera_height: int = 64
    camera_fov: float = 75.
    camera_near: float = .1
    camera_far: float = 60.
    camera_offset: tuple[float, float, float] = (.6, 0., 0.)
    imu_hz: float = 50.
    gps_hz: float = 10.
    imu_noise_std: float = 0.
    gps_noise_std: float = 0.
    imu_bias: tuple[float, ...] = (0.,)*6
    gps_bias: tuple[float, ...] = (0.,)*3
    imu_bias_drift: float = 0.
    update_hz: float = 10.
    range_m: float = 30.
    noise_std: float = 0.

    def validate(self):
        for name in ('lidar_hz', 'lidar_range', 'camera_hz', 'camera_near', 'camera_far'):
            positive(name, getattr(self, name))
        for name in ('lidar_rays', 'camera_width', 'camera_height'):
            positive(name, getattr(self, name), integer=True)
        if not 0 < self.lidar_fov <= 360 or not 0 < self.camera_fov < 180 or self.camera_near >= self.camera_far:
            raise ValueError('Invalid LiDAR/camera projection parameters')
        if not self.lidar_vertical_angles or any(not math.isfinite(x) or abs(x) > 90 for x in self.lidar_vertical_angles):
            raise ValueError('Invalid LiDAR vertical angles')
        if not math.isfinite(self.lidar_noise_std) or self.lidar_noise_std < 0:
            raise ValueError('Invalid LiDAR noise')
        if len(self.camera_offset) != 3 or not all(math.isfinite(x) for x in self.camera_offset):
            raise ValueError('Invalid camera offset')
        positive("imu_hz", self.imu_hz)
        positive("gps_hz", self.gps_hz)
        for name in ('imu_noise_std', 'gps_noise_std', 'imu_bias_drift'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, size in [('imu_bias', 6), ('gps_bias', 3)]:
            values = getattr(self, name)
            if len(values) != size or not all(math.isfinite(x) for x in values):
                raise ValueError(f"{name} must contain {size} finite values")
        positive("sensor update_hz", self.update_hz)
        positive("sensor range_m", self.range_m)
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")


@dataclass
class ObstacleConfig:
    """Reserved settings for future moving obstacles."""
    shapes: tuple[str, ...] = ('box', 'sphere', 'cylinder')
    trajectory: str = 'back_and_forth'
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None
    min_size: float = 1.5
    max_size: float = 3.0
    count: int = 0
    speed: float = 1.

    def validate(self):
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("obstacle count must be a nonnegative integer")
        positive("obstacle speed", self.speed)
        positive("min_size", self.min_size)
        positive("max_size", self.max_size)
        if self.min_size > self.max_size or not self.shapes or any(s not in {'box', 'sphere', 'cylinder'} for s in self.shapes):
            raise ValueError("Invalid obstacle sizes or shapes")
        if self.trajectory not in {'linear', 'waypoint', 'back_and_forth'}:
            raise ValueError("Unknown obstacle trajectory")


@dataclass
class DisturbanceConfig:
    """Reserved world-frame wind acceleration; no disturbance is applied yet."""
    variation_amplitude: tuple[float, float, float] = (0., 0., 0.)
    variation_hz: float = .1
    enabled: bool = False
    wind_acceleration: tuple[float, float, float] = (0., 0., 0.)

    def validate(self):
        positive("variation_hz", self.variation_hz)
        if len(self.variation_amplitude) != 3 or not all(math.isfinite(x) for x in self.variation_amplitude):
            raise ValueError("variation_amplitude must contain three finite values")
        if len(self.wind_acceleration) != 3 or not all(math.isfinite(x) for x in self.wind_acceleration):
            raise ValueError("wind_acceleration must contain three finite values")


@dataclass
class BatteryConfig:
    initial_charge: float = 1.
    hover_rate: float = .0001
    motion_rate: float = .00001
    acceleration_rate: float = .00001
    climb_rate: float = .00002
    terminate_on_depletion: bool = True

    def validate(self):
        if not math.isfinite(self.initial_charge) or not 0 <= self.initial_charge <= 1:
            raise ValueError('initial_charge must be in [0, 1]')
        for name in ('hover_rate', 'motion_rate', 'acceleration_rate', 'climb_rate'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')


@dataclass
class EnvConfig:
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    proximity_distance: float = 3.
    neighbour_count: int = 4
    neighbour_radius: float = 20.
    imu_acceleration_scale: float = 10.
    imu_angular_scale: float = 10.
    scenario: str = "urban"
    num_drones: int = 8
    town: TownConfig = field(default_factory=TownConfig)
    drone: DroneConfig = field(default_factory=DroneConfig)
    gui: bool = True
    low_level_hz: int = 120
    physics_hz: int = 120
    control_hz: int = 20
    max_episode_steps: int = 600
    goal_radius: float = 2.0
    spawn_altitude: float = 12.0
    goal_altitude: float = 12.0
    world_margin: float = 3.0
    realtime: bool = False

    source: RegionConfig = field(default_factory=RegionConfig)
    destination: RegionConfig = field(default_factory=RegionConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    rewards: "RewardWeights" = field(default_factory=lambda: RewardWeights())

    @property
    def physics_dt(self):
        return 1. / self.physics_hz

    @property
    def control_dt(self):
        return 1. / self.control_hz

    def validate(self):
        positive("neighbour_count", self.neighbour_count, integer=True)
        for name in ("neighbour_radius", "imu_acceleration_scale", "imu_angular_scale", "proximity_distance"):
            positive(name, getattr(self, name))
        for name in ("num_drones", "low_level_hz", "physics_hz", "control_hz", "max_episode_steps"):
            positive(name, getattr(self, name), integer=True)
        for name in ("goal_radius", "spawn_altitude", "goal_altitude", "world_margin"):
            positive(name, getattr(self, name))
        for group in (self.battery, self.drone, self.source, self.destination, self.sensors,
                      self.obstacles, self.disturbances, self.rewards):
            group.validate()
        if max(self.sensors.imu_hz, self.sensors.gps_hz, self.sensors.lidar_hz, self.sensors.camera_hz) > self.physics_hz:
            raise ValueError("Sensor frequencies must not exceed physics_hz")
        if self.sensors.lidar_range <= self.drone.radius+1e-4:
            raise ValueError('LiDAR range must exceed collision radius')
        self.town.validate()
        if self.scenario == "indoor":
            self.scenario = "cluttered"
        if self.scenario not in {"urban", "open_field", "cluttered"}:
            raise ValueError("Unknown scenario")
        if self.num_drones < 1:
            raise ValueError('num_drones must be >= 1')
        if self.physics_hz % self.low_level_hz or self.low_level_hz % self.control_hz:
            raise ValueError('physics_hz must divide into low_level_hz, which must divide into control_hz')
        if self.physics_hz <= 0 or self.control_hz <= 0 or self.physics_hz % self.control_hz:
            raise ValueError('physics_hz must be divisible by control_hz')
        if self.max_episode_steps <= 0 or self.goal_radius <= 0:
            raise ValueError('episode length and goal radius must be positive')
        if self.spawn_altitude < 2 or self.goal_altitude < 2:
            raise ValueError('spawn/goal altitudes must be >= 2 m')


@dataclass
class RewardWeights:
    proximity: float = -1.
    energy: float = -1.
    progress: float = .1
    time: float = -.01
    effort: float = -.001
    arrival: float = 20.
    collision: float = -20.



    def validate(self):
        for item in fields(self):
            if not math.isfinite(getattr(self, item.name)):
                raise ValueError(f"Reward weight {item.name} must be finite")
