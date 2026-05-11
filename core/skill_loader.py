"""
Skill Discovery and Loading for Langclaw
"""

import logging
import os

logger = logging.getLogger(__name__)

#===========Data Classes =================

class CategoryMetadata:
    """Parsed CATEGORY.md frontmatter"""
    def __init__(self,name:str,description:str,emoji:str = ''):
        self.name = name
        self.description = description
        self.emoji = emoji
    
class SkillMetadata:
    def __init__(
        self,
        name: str,
        description: str,
        path: str,
        category: str = "",
        emoji: str = "",
        dependencies: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.path = path
        self.category = category
        self.emoji = emoji
        self.dependencies: list[str] = dependencies or []

class Skill:
    """Level 2 - a full loaded skill including its instruction text."""

    def __init__(self, metadata: SkillMetadata, instructions: str) -> None:
        self.metadata = metadata
        self.instructions = instructions

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def description(self) -> str:
        return self.metadata.description

#===========REGISTRY / MAIN LOGIC================

class SkillRegistry:
    def __init__(self,skills_dirs:list[str]|None = None):
        self.skills_dirs: list[str] = list(skills_dirs) if skills_dirs else []
        self._cache: list[SkillMetadata] | None = None
        self._categories: dict[str, CategoryMetadata] = {}

    def invalidate(self)->None:
        self._cache = None
        self._categories = {}

    @property
    def categories(self):
        """Returns discovered category metadata(call discover() first)"""
        if self._cache is None:
            self.discover()
        return self._categories
    
    #--- Level 1 : Metadata discovery ------------

    def discover(self):