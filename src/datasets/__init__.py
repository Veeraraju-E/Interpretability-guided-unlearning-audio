from .adapter_base import DatasetAdapter
from .audiomnist_adapter import AudioMNISTAdapter
from .lahaja_adapter import LahajaAdapter
from .librispeech_adapter import LibriSpeechAdapter
from .speech_commands_adapter import SpeechCommandsAdapter
from .voxceleb_adapter import VoxCelebAdapter

__all__ = ['DatasetAdapter', 'AudioMNISTAdapter', 'LahajaAdapter', 'LibriSpeechAdapter', 'SpeechCommandsAdapter', 'VoxCelebAdapter']

