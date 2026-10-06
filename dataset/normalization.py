TRAIT_RANGE = {
    'liveweight': (29.5, 120.4),
    'backfat': (5.29, 21.43),
}


def normalize(value, trait):
    min_value, max_value = TRAIT_RANGE[trait]
    return (value - min_value) / (max_value - min_value)


def denormalize(value, trait):
    min_value, max_value = TRAIT_RANGE[trait]
    return value * (max_value - min_value) + min_value
