# Hieroglyph Removal Tool

## description:
This repo is a series of tools for automatically removing hieroglyphs from TopBib volumes.

## usage:
Note: this tool uses Pickle files to serialize machine learning models. These are created by makeModel.py:

    python makeModel.py <dataset path>, <model output path>

These models are then used to assist classification and removal of hieroglyphs via:

    python removeGlyphs.py d <image folder path> <image start number> <bracket detection model path> <hieroglyph detection model path>

To create a dataset or add additional data, the tool 'makeDataset.py' can be used. This tool uses an older approach to character detection and should be updated soon, hopefully. It can be run via:

    python makeDataset.py <image folder path> <dataset path> <min page number> <max page number>