DEFAULT_CLUSTER_COUNT = 8
DEFAULT_CORRELATION_THRESHOLD = 0.7
DEFAULT_DROP_CORRELATED = True
DEFAULT_DROP_LOW_VARIANCE = False
DEFAULT_MIN_VARIANCE = 1e-8

# Excludes features that treat lags, that cause multicollinearity and that are not interpretable for the business, such as model frequency features.
ALLOWED_CLUSTER_FEATURES = [
    "n_clicks_observed",
    "n_unique_pages",
    "n_unique_models",
    "n_unique_categories",
    "n_unique_colours",
    "price_mean",
    "price_min",
    "price_max",
    "price_std",
    "high_price_share_first_n",
    "high_price_count_first_n",
    "category_entropy",
    "category_share_blouses",
    "category_share_sale",
    "category_share_skirts",
    "category_share_trousers",
    "top_category_share",
    "mean_model_frequency",
    "max_model_frequency",
    "min_model_frequency",
    "last_page",
    "last_location",
    "first_model_frequency",
    "last_model_frequency",
    "last_category_frequency",
    "last_colour_frequency",
    "category_transition_count",
]
