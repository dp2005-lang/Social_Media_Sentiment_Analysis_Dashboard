TEXT_COLUMN_ALIASES = (
    "text",
    "tweet",
    "full_text",
    "content",
    "comment",
    "review",
    "message",
    "caption",
)


def _normalize_column_name(column):
    return str(column).strip().lower().replace(" ", "_")


def find_text_column(dataframe):
    normalized_columns = {
        _normalize_column_name(column): column for column in dataframe.columns
    }

    for alias in TEXT_COLUMN_ALIASES:
        if alias in normalized_columns:
            return normalized_columns[alias]

    return None


def predict_dataframe(dataframe, *, model, vectorizer, preprocess):
    text_column = find_text_column(dataframe)
    if text_column is None:
        accepted = ", ".join(TEXT_COLUMN_ALIASES)
        raise ValueError(f"CSV needs a text column such as {accepted}.")

    results = dataframe.copy()
    if results.empty:
        results["prediction"] = []
        return results

    texts = results[text_column].fillna("").astype(str)
    cleaned_texts = [preprocess(text) for text in texts]
    features = vectorizer.transform(cleaned_texts)
    results["prediction"] = model.predict(features)
    return results
