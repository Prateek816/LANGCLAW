"""
Module-level convenience functions for Langclaw skill access.
"""

import os

from .models import Skill
from .skill_registry import SkillRegistry


def load_skill_by_name(
    skill_name: str,
    skills_dirs: list[str] | None = None,
) -> Skill | None:
    """Load a skill by name (Level 2)."""
    return SkillRegistry(skills_dirs).load_skill(skill_name)


def search_skills(
    query: str,
    skills_dirs: list[str] | None = None,
) -> list[dict]:
    """Search skills by keyword match in name or description."""
    q = query.lower()
    return [
        {"name": s.name, "description": s.description, "category": s.category}
        for s in SkillRegistry(skills_dirs).discover()
        if q in s.name.lower() or q in s.description.lower()
    ]


def list_skills_in_category(
    category: str,
    skills_dirs: list[str] | None = None,
) -> list[dict]:
    """List skills in a specific category (backward compat)."""
    return [
        {
            "name": s.name,
            "description": s.description,
            "path_name": os.path.basename(s.path),
        }
        for s in SkillRegistry(skills_dirs).discover()
        if s.category == category
    ]