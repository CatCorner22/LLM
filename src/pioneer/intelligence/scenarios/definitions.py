"""Autonomous what-if scenario definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class ScenarioType(StrEnum):
    WEATHER_SHOCK = "weather_shock"
    PIPE_FAILURE = "pipe_failure"
    OCCUPANCY_SURGE = "occupancy_surge"
    MAINTENANCE_DELAY = "maintenance_delay"
    REGULATORY_CHANGE = "regulatory_change"
    SUPPLY_DISRUPTION = "supply_disruption"


class ScenarioParameter(BaseModel):
    name: str
    delta: float = Field(description="Multiplier or additive change applied to baseline")
    unit: str = "ratio"


class ScenarioDefinition(BaseModel):
    id: str
    name: str
    scenario_type: ScenarioType
    description: str
    parameters: list[ScenarioParameter] = Field(default_factory=list)
    duration_days: int = Field(default=30, ge=1)
    probability: float = Field(default=0.1, ge=0.0, le=1.0)


DEFAULT_SCENARIOS: list[ScenarioDefinition] = [
    ScenarioDefinition(
        id="scn_flood",
        name="Severe Flood Event",
        scenario_type=ScenarioType.WEATHER_SHOCK,
        description="River overflow and basement flooding affecting operations",
        parameters=[
            ScenarioParameter(name="flood_risk_index", delta=0.4),
            ScenarioParameter(name="storm_probability_7d", delta=0.3),
        ],
        probability=0.08,
    ),
    ScenarioDefinition(
        id="scn_pipe_burst",
        name="Aged Pipe Burst",
        scenario_type=ScenarioType.PIPE_FAILURE,
        description="Cast-iron main failure due to freeze-thaw cycle",
        parameters=[ScenarioParameter(name="pipe_failure_multiplier", delta=1.5)],
        probability=0.12,
    ),
    ScenarioDefinition(
        id="scn_heat",
        name="Extended Heat Wave",
        scenario_type=ScenarioType.WEATHER_SHOCK,
        description="14-day heat wave stressing HVAC and worker safety",
        parameters=[ScenarioParameter(name="heat_wave_days_forecast", delta=10)],
        probability=0.15,
    ),
    ScenarioDefinition(
        id="scn_occupancy",
        name="Occupancy Surge",
        scenario_type=ScenarioType.OCCUPANCY_SURGE,
        description="Temporary 2x occupancy for event or tenant change",
        parameters=[ScenarioParameter(name="occupancy_multiplier", delta=2.0)],
        probability=0.2,
    ),
    ScenarioDefinition(
        id="scn_maintenance",
        name="Maintenance Backlog",
        scenario_type=ScenarioType.MAINTENANCE_DELAY,
        description="Deferred inspections and repairs for 60 days",
        parameters=[ScenarioParameter(name="maintenance_backlog_days", delta=30)],
        probability=0.18,
    ),
    ScenarioDefinition(
        id="scn_regulatory",
        name="New Safety Mandate",
        scenario_type=ScenarioType.REGULATORY_CHANGE,
        description="Mandatory pipe inspection and building code update",
        parameters=[ScenarioParameter(name="compliance_cost_multiplier", delta=1.3)],
        probability=0.1,
    ),
]
