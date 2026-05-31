from dataclasses import dataclass


@dataclass
class FinalRecommendation:
    verdict: str
    explanation: str


def build_final_recommendation(
    is_recommended: int,
    is_recommended_ml: int,
    is_compatible: int
) -> FinalRecommendation:
    if is_recommended not in (0, 1) or is_recommended_ml not in (0, 1) or is_compatible not in (0, 1):
        raise ValueError("All input values must be 0 or 1.")

    if is_recommended == 1 and is_recommended_ml == 1 and is_compatible == 1:
        return FinalRecommendation(
            verdict="Recomandat",
            explanation=(
                "Produsul este evaluat pozitiv atât de scorul baseline, cât și de modelul ML, "
                "iar compatibilitatea cu profilul utilizatorului este favorabilă."
            )
        )

    if is_recommended == 1 and is_recommended_ml == 1 and is_compatible == 0:
        return FinalRecommendation(
            verdict="Produs bun, dar nepotrivit pentru tine",
            explanation=(
                "Produsul este evaluat pozitiv atât de scorul baseline, cât și de modelul ML, "
                "dar nu este compatibil cu profilul utilizatorului."
            )
        )

    if is_recommended == 0 and is_recommended_ml == 0 and is_compatible == 1:
        return FinalRecommendation(
            verdict="Compatibil cu profilul tău, dar slab evaluat",
            explanation=(
                "Produsul este compatibil cu profilul utilizatorului, "
                "dar nu este susținut nici de scorul baseline, nici de modelul ML."
            )
        )

    if is_recommended == 0 and is_recommended_ml == 0 and is_compatible == 0:
        return FinalRecommendation(
            verdict="Nerecomandat",
            explanation=(
                "Produsul nu este susținut nici de scorul baseline, nici de modelul ML, "
                "și nici nu este compatibil cu profilul utilizatorului."
            )
        )

    if is_compatible == 1:
        return FinalRecommendation(
            verdict="Evaluare incertă, dar compatibil cu profilul tău",
            explanation=(
                "Scorul baseline și modelul ML oferă evaluări diferite asupra produsului. "
                "Compatibilitatea cu profilul utilizatorului este favorabilă, "
                "dar verdictul general rămâne unul incert."
            )
        )

    return FinalRecommendation(
        verdict="Evaluare incertă și nepotrivit pentru tine",
        explanation=(
            "Scorul baseline și modelul ML oferă evaluări diferite asupra produsului, "
            "iar compatibilitatea cu profilul utilizatorului nu este favorabilă."
        )
    )