"""Candidate acquisition boundary for the mock MVP."""

import logging
from pathlib import Path

from src.models import CandidateProfile
from src.sheets_loader import load_candidate_profiles


LOGGER = logging.getLogger(__name__)


def get_candidates(path: Path) -> list[CandidateProfile]:
    """Return candidates from the prepared, manually reviewed public list."""

    candidates = load_candidate_profiles(path)
    LOGGER.info("Loaded %d candidates from %s", len(candidates), path)
    return candidates
