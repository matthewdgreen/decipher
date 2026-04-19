from __future__ import annotations

from models.alphabet import Alphabet
from models.cipher_text import CipherText
from ocr.vision import VisionOCR
from services.claude_api import ClaudeAPI


class OCREngine:
    """Orchestrates OCR processing for both images and text input."""

    def __init__(self, claude_api: ClaudeAPI) -> None:
        self.vision = VisionOCR(claude_api)

    def process_image(self, image_path: str) -> CipherText:
        """Process a cipher image: extract symbols via Vision, build CipherText."""
        symbols, transcription = self.vision.extract_symbols(image_path)

        # Filter out NEWLINE tokens from alphabet (they're structural, not cipher symbols)
        cipher_symbols = [s for s in symbols if s != "NEWLINE"]
        alphabet = Alphabet(cipher_symbols)

        # Remove NEWLINE tokens from transcription for encoding
        clean_transcription = " ".join(
            t for t in transcription.split() if t != "NEWLINE"
        )

        return CipherText(
            raw=clean_transcription,
            alphabet=alphabet,
            source="ocr",
        )

    # Common punctuation characters
    PUNCTUATION = set('.,;:!?\'"-()[]{}…–—/\\@#$%^&*_+=<>~`')

    def process_text(
        self,
        raw_text: str,
        multisym: bool = False,
        space_treatment: str = "word_boundaries",
        punct_treatment: str = "passthrough",
    ) -> CipherText:
        """Process manually entered/pasted ciphertext.

        space_treatment:
          "word_boundaries" - spaces delimit words, excluded from alphabet
          "cipher_symbols"  - spaces are cipher symbols in the alphabet
          "ignore"          - strip all spaces before processing

        punct_treatment:
          "passthrough" - punctuation preserved in text but excluded from cipher alphabet
          "cipher_symbols" - punctuation treated as cipher symbols
          "strip" - remove all punctuation before processing
        """
        ignore: set[str] = set()
        separator: str | None = None

        # Handle spaces
        if space_treatment == "ignore":
            raw_text = raw_text.replace(" ", "")
        elif space_treatment == "word_boundaries":
            ignore.update({" ", "\t", "\n", "\r"})
            separator = " "

        # Handle punctuation
        if punct_treatment == "strip":
            raw_text = "".join(c for c in raw_text if c not in self.PUNCTUATION)
        elif punct_treatment == "passthrough":
            ignore.update(self.PUNCTUATION)

        # Handle newlines — always treat as whitespace unless cipher_symbols
        if space_treatment != "cipher_symbols":
            raw_text = " ".join(raw_text.split())

        alphabet = Alphabet.from_text(raw_text, multisym=multisym, ignore_chars=ignore)
        return CipherText(
            raw=raw_text, alphabet=alphabet, source="manual", separator=separator,
        )
