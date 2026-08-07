"""Sample asset portfolio for demos and benchmarks."""

from pioneer.intelligence.risk.models import (
    AssetPortfolio,
    BuildingProfile,
    GovernanceProfile,
    OperationalProfile,
    PipeProfile,
    WeatherProfile,
)


def sample_portfolio() -> AssetPortfolio:
    """Representative urban commercial property with aging infrastructure."""
    return AssetPortfolio(
        building=BuildingProfile(
            asset_id="BLDG-001",
            year_built=1962,
            occupancy=220,
            floor_area_sqm=4500.0,
            structural_condition=0.62,
            fire_suppression=False,
            seismic_zone=2,
            last_inspection_years_ago=4.5,
        ),
        pipes=[
            PipeProfile(
                asset_id="PIPE-MAIN-A",
                install_year=1958,
                material="cast_iron",
                diameter_mm=200.0,
                pressure_bar=5.5,
                soil_corrosivity=0.55,
                inspection_score=0.45,
            ),
            PipeProfile(
                asset_id="PIPE-SERVICE-B",
                install_year=1988,
                material="ductile_iron",
                diameter_mm=100.0,
                pressure_bar=3.0,
                soil_corrosivity=0.35,
                inspection_score=0.72,
            ),
        ],
        weather=WeatherProfile(
            asset_id="BLDG-001",
            region="Northeast US",
            flood_risk_index=0.42,
            wind_risk_index=0.35,
            heat_wave_days_forecast=3,
            freeze_days_forecast=5,
            storm_probability_7d=0.22,
        ),
        operational=OperationalProfile(
            asset_id="BLDG-001",
            worker_count=18,
            safety_training_hours=6.0,
            prior_incidents_12m=2,
            maintenance_backlog_days=22.0,
            night_shift_ratio=0.25,
        ),
        governance=GovernanceProfile(
            asset_id="BLDG-001",
            conduct_incidents_12m=1,
            policy_training_completion=0.82,
            whistleblower_channel=True,
            sod_conflicts=2,
            role_overlap_ratio=0.28,
            dual_control_gaps=1,
        ),
    )
