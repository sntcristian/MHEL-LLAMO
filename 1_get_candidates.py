from src.retriever import load_disambiguator

def main():
    """Example usage of the EntityDisambiguator."""

    disambiguator = load_disambiguator()

    # Batch example
    texts = [
        "Leonardo da Vinci lived in Firenze",
        "Einstein was born in Germany",
        "Ces hommes qui existent ainsi * (les Chartreux de Rome) sont pourtant les mêmes à qui la guerre et toute son activité suffiraient à peine s'ils s'y étaient accoutumés. C'est un sujet inépuisable de réflexion que ﻿ 3467 les différentes combinaisons de la destinée humaine sur la terre. Il se passe dans l'intérieur de l'ame mille accidents, il se forme mille habitudes qui font de chaque individu un monde et son histoire. Connaître un autre parfaitement serait l'étude d'une vie entière; qu'est-ce donc qu'on entend par connaître les hommes? les gouverner, cela se peut, mais les comprendre, Dieu seul le fait. * Corinne, livre 10. Chap. 1. t. 2. p. 114.",
        "Fermezza di carattere e facoltà di generalizzare formano quelli che si chiamano uomini superiori: essi sanno pensare e sanno operare * : ﻿ 3447 dice M. Say ne' Cenni sugli uomini e la Società."
    ]
    batch_offsets = [[0, 27], [0, 21], [50, 613], [160]]
    batch_lengths = [[17, 7], [8, 7], [4, 7], [31]]

    batch_predictions = disambiguator.get_candidates_batch(texts, batch_offsets, batch_lengths, 10)

    for i, (text, predictions) in enumerate(zip(texts, batch_predictions)):
        print(f"\nText {i + 1}: {text}")
        for pred in predictions:
            print(pred["surface"])
            print(pred["candidates"])


if __name__ == "__main__":
    main()