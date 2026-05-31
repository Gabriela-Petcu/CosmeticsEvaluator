from dataclasses import dataclass


ALLOWED_SKIN_TYPES = {
    "oily",
    "dry",
    "combination",
    "sensitive",
    "normal"
}

ALLOWED_MAIN_CONCERNS = {
    "acne",
    "dehydration",
    "anti_aging",
    "dark_spots",
    "redness",
    "dullness"
}

ALLOWED_BUDGET_LEVELS = {
    "low",
    "medium",
    "high"
}


@dataclass
class UserProfile:
    """
    Represents the user profile used in the user matching module.
 
    Fields are validated at initialization to accept only the values
    officially defined in the project.
 
    Attributes
    skin_type : str
        One of: 'oily', 'dry', 'combination', 'sensitive', 'normal'.
    main_concern : str
        One of: 'acne', 'dehydration', 'anti_aging', 'dark_spots', 'redness', 'dullness'.
    budget_level : str
        One of: 'low', 'medium', 'high'.
    """
    skin_type: str
    main_concern: str
    budget_level: str

    def __post_init__(self):
        if self.skin_type not in ALLOWED_SKIN_TYPES:
            raise ValueError(
                f"Invalid skin_type: '{self.skin_type}'. "
                f"Allowed values are: {sorted(ALLOWED_SKIN_TYPES)}"
            )

        if self.main_concern not in ALLOWED_MAIN_CONCERNS:
            raise ValueError(
                f"Invalid main_concern: '{self.main_concern}'. "
                f"Allowed values are: {sorted(ALLOWED_MAIN_CONCERNS)}"
            )

        if self.budget_level not in ALLOWED_BUDGET_LEVELS:
            raise ValueError(
                f"Invalid budget_level: '{self.budget_level}'. "
                f"Allowed values are: {sorted(ALLOWED_BUDGET_LEVELS)}"
            )