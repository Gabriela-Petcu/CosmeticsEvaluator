from dataclasses import dataclass


@dataclass
class FinalRecommendation:
    verdict: str
    explanation: str


def build_final_recommendation(
    merita: int,
    merita_ml: int,
    se_potriveste: int
) -> FinalRecommendation:
    """
    Construiește verdictul final pe baza celor 3 componente:
    - baseline (Merita)
    - model ML (MeritaML)
    - user matching (SePotriveste)

    Politica folosită în proiect este:

    1. baseline + ML evaluează produsul ca valoare generală
    2. user matching personalizează verdictul pentru utilizator
    3. dacă baseline și ML sunt de acord:
       - rezultatul este considerat stabil
       - user matching doar adaptează verdictul la profil
    4. dacă baseline și ML sunt în conflict:
       - verdictul devine unul prudent, de tip "evaluare incertă"
       - user matching oferă doar context suplimentar

    Astfel:
    - componentele de evaluare generală a produsului sunt baseline și ML
    - componenta de personalizare este user matching
    """

    if merita not in (0, 1) or merita_ml not in (0, 1) or se_potriveste not in (0, 1):
        raise ValueError("Toate valorile trebuie să fie 0 sau 1.")

    # Caz stabil pozitiv: produs bun și potrivit pentru utilizator
    if merita == 1 and merita_ml == 1 and se_potriveste == 1:
        return FinalRecommendation(
            verdict="Recomandat",
            explanation=(
                "Produsul este evaluat pozitiv atât de scorul baseline, cât și de modelul ML, "
                "iar compatibilitatea cu profilul utilizatorului este favorabilă."
            )
        )

    # Caz stabil pozitiv: produs bun în general, dar nu pentru acest utilizator
    if merita == 1 and merita_ml == 1 and se_potriveste == 0:
        return FinalRecommendation(
            verdict="Produs bun, dar nepotrivit pentru tine",
            explanation=(
                "Produsul este evaluat pozitiv atât de scorul baseline, cât și de modelul ML, "
                "dar nu este compatibil cu profilul utilizatorului."
            )
        )

    # Caz stabil negativ: produs slab în general, dar aparent compatibil cu profilul
    if merita == 0 and merita_ml == 0 and se_potriveste == 1:
        return FinalRecommendation(
            verdict="Compatibil cu profilul tău, dar slab evaluat",
            explanation=(
                "Produsul este compatibil cu profilul utilizatorului, "
                "dar nu este susținut nici de scorul baseline, nici de modelul ML."
            )
        )

    # Caz stabil negativ: produs slab și nepotrivit
    if merita == 0 and merita_ml == 0 and se_potriveste == 0:
        return FinalRecommendation(
            verdict="Nerecomandat",
            explanation=(
                "Produsul nu este susținut nici de scorul baseline, nici de modelul ML, "
                "și nici nu este compatibil cu profilul utilizatorului."
            )
        )

    # Cazuri conflictuale baseline vs ML -> verdict prudent
    if se_potriveste == 1:
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