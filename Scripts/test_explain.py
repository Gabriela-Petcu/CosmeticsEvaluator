from Src.explainability import explain_product, print_explanation


sample_product = {
    "n_of_reviews": 1200,
    "n_of_loves": 54000,
    "review_score": 4.6,
    "price_per_ounce": 18.5
}

result = explain_product(sample_product)
print_explanation(result)