import logging
import re
import subprocess
import tempfile
from pathlib import Path

from packaging.version import Version

from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation

logger = logging.getLogger(__name__)


def _is_tool(name: str) -> bool:
    from shutil import which

    return which(name) is not None

espeak_version_pattern = re.compile(r"text-to-speech:\s(?P<version>\d+\.\d+(\.\d+)?)")


def get_espeak_version() -> str:
    output = subprocess.run(["espeak", "--version"], capture_output=True, text=True, check=True).stdout
    match = espeak_version_pattern.search(output)

    return match.group("version")


def get_espeakng_version() -> str:
    output = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True, check=True).stdout
    return output.split()[3]

if _is_tool("espeak-ng"):
    _DEF_ESPEAK_LIB = "espeak-ng"
    _DEF_ESPEAK_VER = get_espeakng_version()
elif _is_tool("espeak"):
    _DEF_ESPEAK_LIB = "espeak"
    _DEF_ESPEAK_VER = get_espeak_version()
else:
    _DEF_ESPEAK_LIB = None
    _DEF_ESPEAK_VER = None


def _espeak_exe(espeak_lib: str, args: list) -> list[str]:
    """Run espeak with the given arguments."""
    cmd = [
        espeak_lib,
        "-q",
        "-b",
        "1",  
    ]
    cmd.extend(args)
    logger.debug("Executing: %s", repr(cmd))

    p = subprocess.run(cmd, capture_output=True, encoding="utf8", check=True)
    for line in p.stderr.strip().split("\n"):
        if line.strip() != "":
            logger.warning("%s: %s", espeak_lib, line.strip())
    res = []
    for line in p.stdout.strip().split("\n"):
        if line.strip() != "":
            logger.debug("%s: %s", espeak_lib, line.strip())
            res.append(line.strip())
    return res


class ESpeak(BasePhonemizer):
    def __init__(
        self,
        language: str,
        punctuations: str = Punctuation.default_puncs(),
        *,
        backend: str | None = None,
        keep_puncs: bool = True,
    ) -> None:
        if _DEF_ESPEAK_LIB is None:
            msg = "[!] No espeak backend found. Install espeak-ng or espeak to your system."
            raise FileNotFoundError(msg)
        self.backend = _DEF_ESPEAK_LIB

        # band-aid for backwards compatibility
        if language == "en":
            language = "en-us"
        if language == "zh-cn":
            language = "cmn"

        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        if backend is not None:
            self.backend = backend

    @property
    def backend(self) -> str:
        return self._ESPEAK_LIB

    @property
    def backend_version(self) -> str:
        return self._ESPEAK_VER

    @backend.setter
    def backend(self, backend: str) -> None:
        if backend not in ["espeak", "espeak-ng"]:
            msg = f"Unknown backend: {backend}"
            raise ValueError(msg)
        self._ESPEAK_LIB = backend
        self._ESPEAK_VER = get_espeakng_version() if backend == "espeak-ng" else get_espeak_version()

    def auto_set_espeak_lib(self) -> None:
        if _is_tool("espeak-ng"):
            self._ESPEAK_LIB = "espeak-ng"
            self._ESPEAK_VER = get_espeakng_version()
        elif _is_tool("espeak"):
            self._ESPEAK_LIB = "espeak"
            self._ESPEAK_VER = get_espeak_version()
        else:
            msg = "Cannot set backend automatically. espeak-ng or espeak not found"
            raise FileNotFoundError(msg)

    @staticmethod
    def name() -> str:
        return "espeak"

    def phonemize_espeak(self, text: str, separator: str = "|", *, tie: bool = False) -> str:
        args = ["-v", f"{self._language}"]
        if tie:
            if self.backend == "espeak":
                args.append("--ipa=1")
            else:
                args.append("--ipa=3")
        else:
            if self.backend == "espeak":
                if Version(self.backend_version) >= Version("1.48.15"):
                    args.append("--ipa=1")
                else:
                    args.append("--ipa=3")
            else:
                args.append("--ipa=1")
        if tie:
            args.append(f"--tie={tie}")

        tmp = tempfile.NamedTemporaryFile(mode="w+t", delete=False, encoding="utf8")
        tmp.write(text)
        tmp.close()
        args.append("-f")
        args.append(tmp.name)
        phonemes = ""
        for line in _espeak_exe(self.backend, args):
            ph_decoded = re.sub(r"\(.+?\)", "", line)

            phonemes += ph_decoded.strip()
        Path(tmp.name).unlink()
        return phonemes.replace("_", separator)

    def _phonemize(self, text: str, separator: str = "") -> str:
        return self.phonemize_espeak(text, separator, tie=False)

    @staticmethod
    def supported_languages() -> dict[str, str]:
        if _DEF_ESPEAK_LIB is None:
            return {}
        args = ["--voices"]
        langs = {}
        for count, line in enumerate(_espeak_exe(_DEF_ESPEAK_LIB, args)):
            if count > 0:
                cols = line.split()
                lang_code = cols[1]
                lang_name = cols[3]
                langs[lang_code] = lang_name
        return langs

    def version(self) -> str:
        return self.backend_version

    @classmethod
    def is_available(cls) -> bool:
        return _is_tool("espeak") or _is_tool("espeak-ng")


if __name__ == "__main__":
    e = ESpeak(language="en-us")
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())

    e = ESpeak(language="en-us", keep_puncs=False)
    print("`" + e.phonemize("hello how are you today?") + "`")

    e = ESpeak(language="en-us", keep_puncs=True)
    print("`" + e.phonemize("hello how are you today?") + "`")
