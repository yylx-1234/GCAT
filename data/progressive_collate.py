"""Collation for asynchronous progressive PASTIS-R prefixes."""

import torch


SATELLITES = ("s2", "s1a", "s1d")


def _pad_2d(sequences):
    lengths = torch.as_tensor([value.shape[0] for value in sequences], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise ValueError("Progressive batches cannot contain empty sequences")
    maximum = int(lengths.max())
    dimension = int(sequences[0].shape[1])
    padded = sequences[0].new_zeros((len(sequences), maximum, dimension))
    mask = torch.zeros((len(sequences), maximum), dtype=torch.bool)
    for index, value in enumerate(sequences):
        length = int(value.shape[0])
        padded[index, :length] = value
        mask[index, :length] = True
    return padded, lengths, mask


def _pad_1d(sequences, maximum):
    padded = torch.zeros((len(sequences), maximum), dtype=torch.long)
    for index, value in enumerate(sequences):
        padded[index, : value.shape[0]] = value.long()
    return padded


def progressive_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty progressive batch")
    output = {}
    modalities = tuple(
        satellite for satellite in SATELLITES if "{}_features".format(satellite) in batch[0]
    )
    for satellite in modalities:
        key = "{}_features".format(satellite)
        padded, lengths, mask = _pad_2d([sample[key] for sample in batch])
        output[key] = padded
        output[satellite] = padded
        output["{}_lengths".format(satellite)] = lengths
        output["{}_mask".format(satellite)] = mask
        output["{}_padding_mask".format(satellite)] = ~mask
        output["{}_dates".format(satellite)] = _pad_1d(
            [sample["{}_dates".format(satellite)] for sample in batch], padded.shape[1]
        )
        output["dates_{}".format(satellite)] = output["{}_dates".format(satellite)]
    if "s2" in modalities:
        quality, quality_lengths, _ = _pad_2d([sample["s2_quality"] for sample in batch])
        if not torch.equal(quality_lengths, output["s2_lengths"]):
            raise ValueError("S2 quality and feature lengths differ")
        output["s2_quality"] = quality

    for key in (
        "label", "source_label", "fold", "patch_id", "parcel_id",
        "instance_id", "pixel_count", "cutoff_date"
    ):
        output[key] = torch.as_tensor([sample[key] for sample in batch], dtype=torch.long)
    output["cutoff"] = [sample["cutoff"] for sample in batch]
    return output


collate_progressive = progressive_collate_fn

