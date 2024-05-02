
import tensorflow as tf

LABEL_KEY = "is_toxic"
FEATURE_KEY = "tweet"


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


regex_patterns = {
    "username": r"@[a-zA-z0-9_]+",
    "retweet": r"rt\s",
    "ampersand": r"&amp",
    "hashtag": r"#[a-zA-Z_\-\d]+",
    "url": r"https?:\/\/\S+|www\.\S+",
    "nonalphanum": r"[^a-zA-Z\d\s+]",
    "numbers": r"\d",
    "multispaces": r"\s+",
}


def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    Args:
        inputs: map from feature keys to raw features.
    Return:
        outputs: map from feature keys to transformed features.
    """

    outputs = {}

    preprocessed_text = tf.strings.lower(inputs[FEATURE_KEY])

    # Remove usernames, rt placeholders, ampersand, hashtags, urls, and unnecessary characters
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["username"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["retweet"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["ampersand"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["hashtag"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["url"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["nonalphanum"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["numbers"], " "
    )
    preprocessed_text = tf.strings.regex_replace(
        preprocessed_text, regex_patterns["multispaces"], " "
    )
    preprocessed_text = tf.strings.strip(preprocessed_text)

    outputs[transformed_name(FEATURE_KEY)] = preprocessed_text
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
