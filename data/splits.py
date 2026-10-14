"""Official PASTIS-R fold handling (never random)."""

OFFICIAL_FOLDS = (1, 2, 3, 4, 5)


def validate_folds(folds):
    values = tuple(int(fold) for fold in folds)
    invalid = sorted(set(values).difference(OFFICIAL_FOLDS))
    if invalid:
        raise ValueError("Invalid official fold(s): {}".format(invalid))
    return values


def folds_for_split(config, split):
    if not config["dataset"].get("use_official_folds", False):
        raise ValueError("Phase 1 requires dataset.use_official_folds=true")
    key = "{}_folds".format(split)
    folds = config.get("data", {}).get(key)
    if folds is None:
        try:
            folds = config["experiment"][key]
        except KeyError:
            raise KeyError("Missing data.{0} or experiment.{0} in config".format(key))
    return validate_folds(folds)


def patch_ids_for_folds(records, folds):
    selected = set(validate_folds(folds))
    return [record.patch_id for record in records if record.fold in selected]
