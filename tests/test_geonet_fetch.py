"""Regression tests for the GeoNet FDSN fetch helpers."""

from pathlib import Path
import sys

from obspy import UTCDateTime
from obspy.core.event import Arrival, Event, Origin, Pick, ResourceIdentifier, WaveformStreamID

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from geonet_fetch import _extract_picks


def test_extract_picks_uses_obspy_arrival_pick_id():
    """GeoNet arrivals reference picks through ObsPy's ``pick_id`` field."""
    pick_id = ResourceIdentifier("smi:test/pick-1")
    pick_time = UTCDateTime("2023-01-01T00:00:00")
    pick = Pick(
        resource_id=pick_id,
        phase_hint="P",
        time=pick_time,
        waveform_id=WaveformStreamID(network_code="NZ", station_code="TEST"),
    )
    arrival = Arrival(pick_id=pick_id, phase="P")
    event = Event(picks=[pick], origins=[Origin(arrivals=[arrival])])

    assert _extract_picks(event) == {"TEST": {"P": pick_time}}


if __name__ == "__main__":
    test_extract_picks_uses_obspy_arrival_pick_id()
    print("PASS test_extract_picks_uses_obspy_arrival_pick_id")
