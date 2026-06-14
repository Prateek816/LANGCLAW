"""
Data classes for Langclaw skill discovery.
"""


class CategoryMetadata:
    """Parsed CATEGORY.md frontmatter"""

    def __init__(self, name: str, description: str, emoji: str = ""):
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