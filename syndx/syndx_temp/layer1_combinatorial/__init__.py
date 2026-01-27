"""
Layer 1: Combinatorial Enumeration - TiTrATE-based archetype generation
"""

from .archetype_generator import ArchetypeGenerator, ClinicalArchetype, TimingPattern, TriggerType, DiagnosisCategory

__all__ = [
    'ArchetypeGenerator',
    'ClinicalArchetype',
    'TimingPattern',
    'TriggerType',
    'DiagnosisCategory'
]
