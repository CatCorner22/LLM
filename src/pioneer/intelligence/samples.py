"""Sample asset portfolio for demos and benchmarks."""

from datetime import UTC, datetime

from pioneer.intelligence.competency_models import (
    CompetencyProfile,
    CompetencySnapshot,
    SkillCompetency,
)
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
        competency=CompetencyProfile(
            asset_id="BLDG-001",
            training_completion_rate=0.95,
            scenario_transfer_score=0.42,
            explanation_audit_score=0.38,
            novel_condition_error_rate=0.48,
            practical_demonstration_rate=0.35,
            certification_only_ratio=0.72,
            assessment_count=3,
            skills=[
                SkillCompetency(
                    skill_id="lockout_tagout",
                    role="maintenance",
                    criticality=0.95,
                    training_completion_rate=0.98,
                    scenario_transfer_score=0.35,
                    explanation_audit_score=0.32,
                    novel_condition_error_rate=0.55,
                    practical_demonstration_rate=0.28,
                    certification_only_ratio=0.82,
                    assessment_count=2,
                ),
                SkillCompetency(
                    skill_id="confined_space",
                    role="operations",
                    criticality=0.9,
                    training_completion_rate=0.92,
                    scenario_transfer_score=0.78,
                    explanation_audit_score=0.81,
                    novel_condition_error_rate=0.14,
                    practical_demonstration_rate=0.85,
                    certification_only_ratio=0.18,
                    assessment_count=4,
                ),
                SkillCompetency(
                    skill_id="hazmat_handling",
                    role="facilities",
                    criticality=0.85,
                    training_completion_rate=0.94,
                    scenario_transfer_score=0.48,
                    explanation_audit_score=0.41,
                    novel_condition_error_rate=0.42,
                    practical_demonstration_rate=0.38,
                    certification_only_ratio=0.68,
                    assessment_count=3,
                ),
            ],
            history=[
                CompetencySnapshot(
                    captured_at=datetime(2025, 6, 1, tzinfo=UTC),
                    training_completion_rate=0.88,
                    scenario_transfer_score=0.58,
                    explanation_audit_score=0.52,
                    novel_condition_error_rate=0.32,
                    practical_demonstration_rate=0.48,
                ),
                CompetencySnapshot(
                    captured_at=datetime(2025, 12, 1, tzinfo=UTC),
                    training_completion_rate=0.92,
                    scenario_transfer_score=0.5,
                    explanation_audit_score=0.45,
                    novel_condition_error_rate=0.4,
                    practical_demonstration_rate=0.41,
                ),
                CompetencySnapshot(
                    captured_at=datetime(2026, 2, 1, tzinfo=UTC),
                    training_completion_rate=0.95,
                    scenario_transfer_score=0.42,
                    explanation_audit_score=0.38,
                    novel_condition_error_rate=0.48,
                    practical_demonstration_rate=0.35,
                ),
            ],
        ),
    )


def expert_competency_profile(asset_id: str = "EXP-DEMO") -> CompetencyProfile:
    """Verified expert profile for benchmark comparisons."""
    return CompetencyProfile(
        asset_id=asset_id,
        training_completion_rate=0.96,
        scenario_transfer_score=0.9,
        explanation_audit_score=0.88,
        novel_condition_error_rate=0.07,
        practical_demonstration_rate=0.93,
        certification_only_ratio=0.12,
        assessment_count=5,
    )
