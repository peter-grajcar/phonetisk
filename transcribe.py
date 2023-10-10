#!/usr/bin/env python3
import re
import sys
from collections import defaultdict
import ufal.morphodita as morphodita


RULE_REGEX = re.compile(
    r"\s*((?P<flags>(\$\S+,)*\$\S+)\s+)?((?P<left>\S+)\))?\s*(?P<source>\S+)\s*(\((?P<right>\S+))?\s*->\s*(?P<targets>\S+)\s*"
)

UNVOICED_GROUP = "F"
VOICED_GROUP = "G"
VOICED_OBSTRUENT_GROUP = "Y"
VOWEL_GROUP = "A"
NOT_VOWEL_GROUP = "K"
SPACE = "_"

UNVOICED = ["p", "t", "ť", "k", "ch", "c", "č", "s", "š", "f", "f"]
VOICED = ["b", "d", "ď", "g", "h", "dz", "dž", "z", "ž", "v", "w"]
VOWELS = ["a", "e", "i", "o", "u", "y", "á", "é", "í", "ó", "ú", "ý", "ô"]
SONORANTS = ["r", "ŕ", "n", "m", "l", "ĺ", "ľ", "j"]

UNVOICED_PHONES = ["p", "t", "c", "k", "x", "G", "ts", "tS", "s", "S", "f", "c:", "s:" "S:" "ts:", "tS:"]
VOICED_PHONES =   ["b", "d", "J\\", "g", "h", "h\\", "dz", "dZ", "z", "Z", "v", "J\\:", "z:" "Z\\:", "dz:", "dZ:"]
VOWEL_PHONES = ["a:", "E:", "i:", "o:", "u:", "y:", "a", "{", "E", "i", "O", "U", "U_^"]
DIPHTHONGS = ["i_^a", "i_^E", "i_^u", "U_^O"]
SONORANT_PHONES = ["l:", "l=:", "r:", "r=:", "r", "l", "l_j", "L", "n", "J", "m", "j"]

MORPHODITA_MORPHO_MODEL = "models/slovak-morfflex-pdt-170914/slovak-morfflex-170914.dict"
MORPHODITA_TAGGER_MODEL = "models/slovak-morfflex-pdt-170914/slovak-morfflex-pdt-170914.tagger"

SEPARATOR = " "


def add_voicing(phone: str) -> str:
    try:
        return VOICED_PHONES[UNVOICED_PHONES.index(phone)]
    except ValueError:
        return phone


def remove_voicing(phone: str) -> str:
    try:
        return UNVOICED_PHONES[VOICED_PHONES.index(phone)]
    except ValueError:
        return phone


def change_voicing(phone: str) -> str:
    if phone in UNVOICED_PHONES:
        return add_voicing(phone)
    elif phone in VOICED_PHONES:
        return remove_voicing(phone)
    else:
        return phone


class Rule:
    def __init__(self, left: str, source: str, right: str, targets: str, flags: str):
        self.left = left
        self.source = source
        self.right = right
        self.targets = targets
        self.flags = flags
        self._construct_regex()

    @property
    def specificity(self) -> tuple[int, int, int]:
        ctx = 0
        if self.right:
            ctx += sum(1 if ch.isupper() else 2 for ch in self.right)
        if self.left:
            ctx += sum(1 if ch.isupper() else 2 for ch in self.left)
        return (len(self.source), len(self.flags), ctx)

    def _substitute_groups(self, s: str) -> str:
        return (
            s.replace(SPACE, " ")
            .replace(UNVOICED_GROUP, "(?:" + "|".join(UNVOICED) + ")")
            .replace(VOICED_OBSTRUENT_GROUP, "(?:" + "|".join(VOICED) + ")")
            .replace(VOICED_GROUP, "(?:" + "|".join(VOICED + VOWELS + SONORANTS) + ")")
            .replace(VOWEL_GROUP, "[" + "".join(VOWELS) + "]")
            .replace(NOT_VOWEL_GROUP, "[^" + "".join(VOWELS) + "]")
        )

    def _construct_regex(self):
        left = self._substitute_groups(self.left[::-1]) if self.left else ""
        right = self._substitute_groups(self.right) if self.right else ""
        self._left_regex = re.compile(f"^{left}", re.IGNORECASE) if self.left else None
        self._right_regex = re.compile(f"^{self.source}{right}", re.IGNORECASE)

    def match(self, text: str, idx: int, flags: list[str] = []) -> bool:
        left_ctx = text[:idx][::-1]
        right_ctx = text[idx:]
        if self.flags and not all(flag in flags for flag in self.flags):
            return False
        if self._left_regex and not re.match(self._left_regex, left_ctx):
            return False
        if m := re.match(self._right_regex, right_ctx):
            return True
        return False

    def __repr__(self) -> str:
        return f"{self.flags} {self.left + ') ' if self.left else ''}{self.source}{' (' + self.right if self.right else ''} -> {self.targets}"


def transcribe(rules: list[Rule], flags: list[tuple[int, int, set[str]]], text: str):
    text = " " + text + " "
    transcription = []
    n = len(text)
    i = 0
    flag_idx = 0
    while i < n:
        if flag_idx < len(flags) and flags[flag_idx][1] <= i:
            flag_idx += 1

        if text[i] in {" ", "-", ","}:
            i += 1
            continue

        for rule in rules:
            if rule.match(text, i, flags[flag_idx][2] if flag_idx < len(flags) else []):
                if rule.targets[0] != "_":
                    transcription.extend(rule.targets)
                i += len(rule.source)
                break
        else:
            print(
                f"Could not transcribe '{text}'. No rule for '{text[i]}'.",
                file=sys.stderr,
            )
            return []

    transcription = apply_regressive_assimilation(transcription)
    return " ".join(transcription)


def tag_word(text: str, morpho: morphodita.Morpho) -> list[list[str]]:
    forms = morphodita.Forms()
    lemmas = morphodita.TaggedLemmas()
    tokens = morphodita.TokenRanges()
    tokenizer = tagger.newTokenizer()
    tokenizer.setText(text)

    tags = []
    while tokenizer.nextSentence(forms, tokens):
        for token in tokens:
            morpho.analyze(text[token.start : token.start + token.length], morpho.GUESSER, lemmas)
            possible_tags = []
            for lemma in lemmas:
                tags.append((token.start, token.start + token.length, lemma.tag))

    return tags


def tag_sentence(text: str, tagger: morphodita.Tagger) -> list[tuple[int, int, str]]:
    forms = morphodita.Forms()
    lemmas = morphodita.TaggedLemmas()
    tokens = morphodita.TokenRanges()
    tokenizer = tagger.newTokenizer()
    tokenizer.setText(text)

    tags = []
    while tokenizer.nextSentence(forms, tokens):
        tagger.tag(forms, lemmas)
        for i, (form, lemma) in enumerate(zip(forms, lemmas)):
            token = tokens[i]
            tags.append((token.start, token.start + token.length, lemma.tag))

    return tags


def create_flags(tag: str) -> tuple[int, int, set[str]]:
    flags = set()

    if tag[0] == "V":
        flags.add("$verb")
    elif tag[0] == "N":
        flags.add("$noun")
    elif tag[0] == "A":
        flags.add("$adj")
    elif tag[0] == "D":
        flags.add("$adv")
    elif tag[0] == "C":
        if tag[1] == "r":
            flags.add("$num_ord")
        elif tag[1] == "l":
            flags.add("$num_card")

    if tag[3] == "P":
        flags.add("$pl")
    elif tag[3] == "S":
        flags.add("$sg")

    return flags


def devoice_final(phones: list[str]) -> list[str]:
    if phones[-1] in VOICED_PHONES:
        phone_set = VOICED_PHONES
    elif phones[-1] in UNVOICED_PHONES:
        phone_set = UNVOICED_PHONES
    else:
        return phones
    i = len(phones) - 1
    while i >= 0 and phones[i] in phone_set:
        phones[i] = change_voicing(phones[i])
        i -= 1

    return phones


def apply_regressive_assimilation(phones: list[str]) -> list[str]:
    new_phones = phones.copy()
    i = len(new_phones) - 1
    voicing_f = None
    while i > 0:
        if voicing_f:
            new_phones[i] = voicing_f(new_phones[i])

        if new_phones[i] in VOICED_PHONES:
            voicing_f = add_voicing
        elif new_phones[i] in UNVOICED_PHONES:
            voicing_f = remove_voicing
        else:
            voicing_f = None

        i -= 1

    return new_phones


if __name__ == "__main__":
    morpho = morphodita.Morpho.load(MORPHODITA_MORPHO_MODEL)
    tagger = morphodita.Tagger.load(MORPHODITA_TAGGER_MODEL)

    rules = []
    with open("rules") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if m := re.match(RULE_REGEX, line):
                groups = m.groupdict()
                rules.append(
                    Rule(
                        groups["left"],
                        groups["source"],
                        groups["right"],
                        groups["targets"].split(","),
                        groups["flags"].split(",") if groups["flags"] else [],
                    )
                )
            else:
                print(f"Could not parse rule '{line}'", file=sys.stderr)

    with open("dict") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            word, transcription = line.split("\t", maxsplit=1)
            rules.append(
                Rule("_", word, "_", transcription.split(" "), [])
            )

    rules.sort(key=lambda rule: rule.specificity, reverse=True)
    # for rule in rules:
    #   print(rule, rule.specificity)

    for word in sys.stdin:
        word = word.strip()
        is_sentece = len(word.split(" ")) > 1

        if is_sentece:
            tags = tag_sentence(word, tagger)
            flags = [(start, end, create_flags(tag)) for start, end, tag in tags]
            transcription = transcribe(rules, flags, word)
            print(word, transcription, sep=SEPARATOR)
        else:
            tags = tag_word(word, morpho)
            transcriptions = set()
            for start, end, tag in tags:
                flags = [(start, end, create_flags(tag))]
                transcription = transcribe(rules, flags, word)
                if not transcription:
                    continue
                transcriptions.add(transcription)
                devoiced = devoice_final(transcription.split(" "))
                transcriptions.add(" ".join(devoiced))

            for transcription in transcriptions:
                print(word, transcription, sep=SEPARATOR)
