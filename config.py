"""Paths, study area, bands and the glider registry."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RODEO = ROOT.parent / "hackathon-data"
DATA, ENV, OUT, WEB = ROOT / "data", ROOT / "data" / "env", ROOT / "out", ROOT / "web"
for _d in (DATA, ENV, OUT):
    _d.mkdir(parents=True, exist_ok=True)

ERDDAP = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap"

# Rodeo box off Kaena Point, plus a margin so edge positions still find a model cell
BOX = dict(lat_min=21.10, lat_max=21.65, lon_min=-158.50, lon_max=-158.05)
CENTRE = (21.384, -158.313)
MISSION_DAYS = ("2026-01-27", "2026-02-13")     # [start, end)

SURFACE_DEPTH_M = 25.0      # above this, glider self-noise dominates

# Target bands: one per Rodeo call type (species.yaml equivalent, kept inline for simplicity)
BANDS = {
    "FIWH": dict(name="Fin whale", call="20 Hz pulse", lo=10, hi=30, centre=20, sl_db=189),
    "HUWH": dict(name="Humpback whale", call="song", lo=200, hi=2000, centre=700, sl_db=175),
    "SPWH": dict(name="Sperm whale", call="usual clicks", lo=5000, hi=15000, centre=10000, sl_db=230),
    "UNDO": dict(name="Odontocete", call="whistles", lo=5000, hi=20000, centre=12000, sl_db=160),
    "BBWH": dict(name="Beaked whale", call="FM upsweep", lo=25000, hi=50000, centre=38000, sl_db=220),
}

# Features the model sees. `forecastable` marks those available in a PacIOOS forecast.
FEATURES = [
    ("wind_ms", True), ("rain_mmhr", True), ("hs_m", True), ("tp_s", True), ("swell_m", True),
    ("cur_ms", True), ("temp_c", True), ("seafloor_m", False),
    ("hour_sin", True), ("hour_cos", True),
    ("depth_m", False), ("depth_rate_ms", False), ("frac_surface", False),
]
FEATURE_NAMES = [f for f, _ in FEATURES]

GLIDERS = {
    "capex987": dict(platform="Slocum G3S", fs_khz=512,
        h5="noise data/capex987_20260128/capex987_20260128.h5",
        science="capex987_20260128/capex987_20260128_science_timeseries.csv",
        cols=dict(time="time_utc", depth="depth", lat="latitude", lon="longitude"),
        eng=dict(file="capex987_20260128/capex987-20260128_flight_timeseries_engineering.csv",
                 time="time_utc", depth="m_depth"),
        gps="capex987_20260128/capex987-20260128_GPS_timeseries.csv", gps_format="start_end"),
    "stenella": dict(platform="Slocum G3S", fs_khz=200,
        h5="noise data/stenella-20260128/stenella-20260128.h5",
        science="stenella_20260128/stenella-20260128_science_timeseries.csv",
        cols=dict(time="time_utc", depth="depth", lat="latitude", lon="longitude"),
        eng=dict(file="stenella_20260128/stenella-20260128_flight_timeseries_engineering.csv",
                 time="time_utc", depth="m_depth"),
        gps="stenella_20260128/stenella-20260128_GPS_timeseries.csv", gps_format="start_end"),
    "risso": dict(platform="Slocum G3S", fs_khz=2,
        h5="noise data/risso-20260128/risso-20260128.h5",
        science="risso_20260128/risso-20260128_science_timeseries.csv",
        cols=dict(time="time_utc", depth="depth", lat="latitude", lon="longitude"),
        eng=dict(file="risso_20260128/risso-20260128_flight_timeseries_engineering.csv",
                 time="time_utc", depth="m_depth"),
        gps="risso_20260128/risso-20260128_GPS_timeseries.csv", gps_format="start_end"),
    "sg607": dict(platform="Seaglider M1", fs_khz=200,
        h5="noise data/sg607_20260128/sg607_20260128.h5",
        science="sg607_20260128/sg607_20260128_science_timeseries.csv",
        cols=dict(time="time", depth="depth", lat="latitude", lon="longitude"),
        gps="sg607_20260128/sg607_20260128_GPS_timeseries.csv", gps_format="seaglider"),
    "sg274": dict(platform="Seaglider SGX", fs_khz=200,
        h5="noise data/sg274_20260128/sg274_20260128.h5",
        science="sg274_20260128/sg274_20260128_science_timeseries.csv",
        cols=dict(time="time", depth="depth", lat="latitude", lon="longitude"),
        gps="sg274_20260128/sg274_20260128_GPS_timeseries.csv", gps_format="seaglider"),
    "SEA117": dict(platform="SeaExplorer", fs_khz=96,
        h5="noise data/SEA117-M026_20260128_30sec/SEA117-M026_20260128_30sec.h5",
        science="SEA117-M026_20260128/SEA117-M026_20260128_science_timeseries.csv",
        cols=dict(time="Time_UTC", depth="NAV_DEPTH", lat="NAV_LATITUDE", lon="NAV_LONGITUDE"),
        eng=dict(file="SEA117-M026_20260128/SEA117-M026_20260128_flight_timeseries_engineering.csv",
                 time="Time_UTC", depth="Depth", lat="Lat", lon="Lon"),
        gps="SEA117-M026_20260128/SEA117-M026_20260128_GPS_timeseries.csv", gps_format="point"),
    "belladonna": dict(platform="Oceanscout", fs_khz=200, exclude=True,
        exclude_reason="GPS fix 2,120 km from the box, salinity 1.81-40.08 PSU, "
                       "acoustics uncorrelated with the fleet",
        h5="noise data/belladonna_20260128/belladonna_20260128.h5",
        science="belladonna_20260128/Belladonna_2026-01-28_science.csv",
        cols=dict(time="time", depth="depth_m", lat="latitude", lon="longitude"),
        gps="belladonna_20260128/Belladonna_2026-01-28_GPS.csv", gps_format="point"),
}

ACTIVE = {k: v for k, v in GLIDERS.items() if not v.get("exclude")}
