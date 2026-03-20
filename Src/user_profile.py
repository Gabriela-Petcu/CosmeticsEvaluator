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
    Reprezintă profilul utilizatorului folosit în modulul de user matching.

    Câmpurile sunt validate la inițializare pentru a accepta doar
    valorile definite oficial în proiect.
    """
    skin_type: str
    main_concern: str
    budget_level: str

    def __post_init__(self):
        if self.skin_type not in ALLOWED_SKIN_TYPES:
            raise ValueError(
                f"skin_type invalid: '{self.skin_type}'. "
                f"Valorile permise sunt: {sorted(ALLOWED_SKIN_TYPES)}"
            )

        if self.main_concern not in ALLOWED_MAIN_CONCERNS:
            raise ValueError(
                f"main_concern invalid: '{self.main_concern}'. "
                f"Valorile permise sunt: {sorted(ALLOWED_MAIN_CONCERNS)}"
            )

        if self.budget_level not in ALLOWED_BUDGET_LEVELS:
            raise ValueError(
                f"budget_level invalid: '{self.budget_level}'. "
                f"Valorile permise sunt: {sorted(ALLOWED_BUDGET_LEVELS)}"
            )