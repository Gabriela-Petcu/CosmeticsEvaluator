from dataclasses import dataclass


@dataclass
class UserProfile:
    skin_type: str
    main_concern: str
    budget_level: str