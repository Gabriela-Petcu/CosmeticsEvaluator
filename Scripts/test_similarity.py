from Src.io import load_skincare_dv
from Src.similarity import get_top_similar_products

def main():
    df = load_skincare_dv()

    result = get_top_similar_products(
        df=df,
        product_name="CC+ Cream Oil-Free Matte with SPF 40",
        top_n=5
    )

    print(result.to_string(index=False))

if __name__ == "__main__":
    main()