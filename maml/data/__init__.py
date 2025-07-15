from ._base import MultiAnnotatorDataset, SSLDatasetWrapper
from ._dopanim import Dopanim
from ._label_me import LabelMe
from ._music_genres import MusicGenres
from ._sentiment_polarity import SentimentPolarity
from ._reuters import Reuters


__all__ = [
    "MultiAnnotatorDataset",
    "SSLDatasetWrapper",
    "Dopanim",
    "LabelMe",
    "MusicGenres",
    "SentimentPolarity",
    "Reuters",
]
