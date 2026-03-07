from dataclasses import dataclass


@dataclass
class FinalRecommendation:
    verdict: str
    explanation: str


def build_final_recommendation(merita: int, se_potriveste: int) -> FinalRecommendation:
    if merita == 1 and se_potriveste == 1:
        return FinalRecommendation(
            verdict="Recomandat",
            explanation="Produsul merită în general și este compatibil cu profilul utilizatorului."
        )

    if merita == 1 and se_potriveste == 0:
        return FinalRecommendation(
            verdict="Merită, dar nu este potrivit pentru tine",
            explanation="Produsul este bine evaluat în general, dar nu se potrivește profilului utilizatorului."
        )

    if merita == 0 and se_potriveste == 1:
        return FinalRecommendation(
            verdict="Ți s-ar potrivi, dar nu merită suficient",
            explanation="Produsul este relativ compatibil cu profilul utilizatorului, dar nu are un verdict general suficient de bun."
        )

    return FinalRecommendation(
        verdict="Nerecomandat",
        explanation="Produsul nu merită în general și nici nu este compatibil cu profilul utilizatorului."
    )