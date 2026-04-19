from __future__ import annotations

from services.claude_api import ClaudeAPI

SYMBOL_EXTRACTION_PROMPT = """\
You are analyzing an image of ciphertext that may contain unusual or invented symbols.

Your task:
1. Identify every unique symbol in the image.
2. Assign each unique symbol a short, consistent label. If the symbol is a recognizable
   character (letter, number, punctuation), use that character. Otherwise, assign a
   descriptive label like SYM_1, SYM_2, etc.
3. Transcribe the full ciphertext as a sequence of these labels, separated by spaces.
   Preserve line breaks as NEWLINE tokens.

Output format:
ALPHABET: <comma-separated list of all symbol labels in order of first appearance>
TEXT:
<space-separated transcription>

Example output:
ALPHABET: A,B,SYM_1,SYM_2,C
TEXT:
A SYM_1 B SYM_2 C A NEWLINE B SYM_1 A C SYM_2
"""


class VisionOCR:
    """Extract symbols from cipher images using Claude Vision."""

    def __init__(self, claude_api: ClaudeAPI) -> None:
        self.api = claude_api

    def extract_symbols(self, image_path: str) -> tuple[list[str], str]:
        """Send an image to Claude Vision for symbol extraction.

        Returns:
            (symbol_list, transcription) where transcription is
            space-separated symbol labels.
        """
        response = self.api.vision_request(image_path, SYMBOL_EXTRACTION_PROMPT)
        return self._parse_response(response)

    @staticmethod
    def _parse_response(response: str) -> tuple[list[str], str]:
        """Parse the structured response from Claude Vision."""
        symbols: list[str] = []
        text_lines: list[str] = []
        in_text = False

        for line in response.splitlines():
            line = line.strip()
            if line.startswith("ALPHABET:"):
                alpha_str = line[len("ALPHABET:"):].strip()
                symbols = [s.strip() for s in alpha_str.split(",") if s.strip()]
            elif line.startswith("TEXT:"):
                in_text = True
            elif in_text and line:
                text_lines.append(line)

        transcription = " ".join(text_lines)
        # Replace NEWLINE tokens with actual newlines for display,
        # but keep them as tokens in the transcription
        return symbols, transcription
