#include "mtl_lang_preprocess.h"

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace tts_cpp::chatterbox::detail {

namespace {

// UTF-8 Helper
bool utf8_decode(const char * s, size_t len, size_t & pos, uint32_t & cp) {
    if (pos >= len) {
        return false;
    }
    uint8_t b0 = (uint8_t) s[pos];
    if (b0 < 0x80) {
        cp = b0;
        pos += 1;
        return true;
    }
    int extra;
    if ((b0 & 0xE0) == 0xC0) { cp = b0 & 0x1F; extra = 1; }
    else if ((b0 & 0xF0) == 0xE0) { cp = b0 & 0x0F; extra = 2; }
    else if ((b0 & 0xF8) == 0xF0) { cp = b0 & 0x07; extra = 3; }
    else { cp = 0xFFFD; pos += 1; return true; }
    if (pos + 1 + extra > len) { cp = 0xFFFD; pos += 1; return true; }
    for (int i = 0; i < extra; ++i) {
        uint8_t b = (uint8_t) s[pos + 1 + i];
        if ((b & 0xC0) != 0x80) { cp = 0xFFFD; pos += 1; return true; }
        cp = (cp << 6) | (b & 0x3F);
    }
    pos += 1 + extra;
    return true;
}

void utf8_append(uint32_t cp, std::string & out) {
    if (cp < 0x80) {
        out.push_back((char) cp);
    } else if (cp < 0x800) {
        out.push_back((char) (0xC0 | (cp >> 6)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back((char) (0xE0 | (cp >> 12)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cp >> 18)));
        out.push_back((char) (0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cp & 0x3F)));
    }
}

// ─── HINDI (hi) ──────────────────────────────────────────────────────────────

std::string get_devanagari_vowel(uint32_t cp) {
    switch (cp) {
        case 0x0905: return "a";
        case 0x0906: return "aa";
        case 0x0907: return "i";
        case 0x0908: return "ii";
        case 0x0909: return "u";
        case 0x090A: return "uu";
        case 0x090B: return "r";
        case 0x090F: return "e";
        case 0x0910: return "ai";
        case 0x0913: return "o";
        case 0x0914: return "au";
        default: return "";
    }
}

std::string get_devanagari_consonant(uint32_t cp) {
    switch (cp) {
        case 0x0915: return "k";
        case 0x0916: return "kh";
        case 0x0917: return "g";
        case 0x0918: return "gh";
        case 0x0919: return "ng";
        case 0x091A: return "c";
        case 0x091B: return "ch";
        case 0x091C: return "j";
        case 0x091D: return "jh";
        case 0x091E: return "ny";
        case 0x091F: return "t";
        case 0x0920: return "th";
        case 0x0921: return "d";
        case 0x0922: return "dh";
        case 0x0923: return "n";
        case 0x0924: return "t";
        case 0x0925: return "th";
        case 0x0926: return "d";
        case 0x0927: return "dh";
        case 0x0928: return "n";
        case 0x092A: return "p";
        case 0x092B: return "ph";
        case 0x092C: return "b";
        case 0x092D: return "bh";
        case 0x092E: return "m";
        case 0x092F: return "y";
        case 0x0930: return "r";
        case 0x0931: return "r";
        case 0x0932: return "l";
        case 0x0933: return "l";
        case 0x0935: return "v";
        case 0x0936: return "sh";
        case 0x0937: return "sh";
        case 0x0938: return "s";
        case 0x0939: return "h";
        default: return "";
    }
}

std::string get_devanagari_matra(uint32_t cp) {
    switch (cp) {
        case 0x093E: return "aa";
        case 0x093F: return "i";
        case 0x0940: return "ii";
        case 0x0941: return "u";
        case 0x0942: return "uu";
        case 0x0943: return "r";
        case 0x0947: return "e";
        case 0x0948: return "ai";
        case 0x094B: return "o";
        case 0x094C: return "au";
        default: return "";
    }
}

std::string devanagari_to_roman(const std::string& text) {
    std::vector<uint32_t> cps;
    size_t pos = 0;
    uint32_t cp;
    while (utf8_decode(text.data(), text.size(), pos, cp)) {
        cps.push_back(cp);
    }

    std::string out;
    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t current = cps[i];
        
        std::string vow = get_devanagari_vowel(current);
        if (!vow.empty()) {
            out += vow;
            continue;
        }

        std::string cons = get_devanagari_consonant(current);
        if (!cons.empty()) {
            out += cons;
            
            // Check next character
            bool has_vowel_modifier = false;
            if (i + 1 < cps.size()) {
                uint32_t next = cps[i + 1];
                std::string matra = get_devanagari_matra(next);
                if (!matra.empty()) {
                    out += matra;
                    i++; // skip next since we consumed it
                    has_vowel_modifier = true;
                } else if (next == 0x094D) { // virama / halant
                    // suppresses inherent vowel
                    i++; // skip next
                    has_vowel_modifier = true;
                }
            }
            if (!has_vowel_modifier) {
                // Word-final schwa deletion: do not append inherent "a" at the end of a word
                bool is_end_of_word = true;
                if (i + 1 < cps.size()) {
                    uint32_t next = cps[i + 1];
                    // If next codepoint is within the Devanagari block, it's not the end of the word
                    if ((next >= 0x0900 && next <= 0x094F) || (next >= 0x0958 && next <= 0x097F)) {
                        is_end_of_word = false;
                    }
                }
                if (!is_end_of_word) {
                    out += "a";
                }
            }
            continue;

        }

        // Marks
        if (current == 0x0901 || current == 0x0902) { // chandrabindu / anusvara
            out += "n";
        } else if (current == 0x0903) { // visarga
            out += "h";
        } else if (current == 0x093C || current == 0x094D) {
            // Nukta/Halant alone - skip/do nothing
        } else {
            // Pass through punctuation/non-Devanagari
            utf8_append(current, out);
        }
    }
    return out;
}

// ─── JAPANESE (ja) ───────────────────────────────────────────────────────────

const char* KANA_ROMAN_TABLE[] = {
    "a", "a", "i", "i", "u", "u", "e", "e", "o", "o", // ぁ, あ, ぃ, い, ぅ, う, ぇ, え, ぉ, お (0x3041-0x304A)
    "ka", "ga", "ki", "gi", "ku", "gu", "ke", "ge", "ko", "go", // か to ご (0x304B-0x3054)
    "sa", "za", "shi", "ji", "su", "zu", "se", "ze", "so", "zo", // さ to ぞ (0x3055-0x305E)
    "ta", "da", "chi", "ji", "tsu", "tsu", "tsu", "de", "te", "de", "to", "do", // た to ど (0x305F-0x3069) (includes small tsu, etc.)
    "na", "ni", "nu", "ne", "no", // な to の (0x306A-0x306E)
    "ha", "ba", "pa", "hi", "bi", "pi", "fu", "bu", "pu", "he", "be", "pe", "ho", "bo", "po", // は to ぽ (0x306F-0x307D)
    "ma", "mi", "mu", "me", "mo", // ま to も (0x307E-0x3082)
    "ya", "ya", "yu", "yu", "yo", "yo", // ゃ, や, ゅ, ゆ, ょ, よ (0x3083-0x3088)
    "ra", "ri", "ru", "re", "ro", // ら to ろ (0x3089-0x308D)
    "wa", "wa", "i", "e", "wo", "n" // ゎ, わ, ゐ, ゑ, を, ん (0x308E-0x3093)
};

std::string kana_to_romaji(const std::string& text) {
    std::vector<uint32_t> cps;
    size_t pos = 0;
    uint32_t cp;
    while (utf8_decode(text.data(), text.size(), pos, cp)) {
        cps.push_back(cp);
    }

    std::string out;
    bool double_next = false;

    for (size_t i = 0; i < cps.size(); ++i) {
        uint32_t current = cps[i];

        // Kanji check: warn + replace with space
        if (current >= 0x4E00 && current <= 0x9FFF) {
            fprintf(stderr, "mtl_tokenizer [ja]: kanji character U+%04X replaced with space\n", current);
            out += " ";
            continue;
        }

        // Normalize Katakana to Hiragana range for lookup
        uint32_t norm = current;
        if (current >= 0x30A1 && current <= 0x30F3) {
            norm = current - 0x30A0 + 0x3040;
        }

        if (norm >= 0x3041 && norm <= 0x3093) {
            size_t idx = norm - 0x3041;
            
            // Special handling for small tsu (ッ/っ)
            if (norm == 0x3063) {
                double_next = true;
                continue;
            }

            std::string romaji = KANA_ROMAN_TABLE[idx];
            
            // Check next char for small ya/yu/yo
            if (i + 1 < cps.size()) {
                uint32_t next_norm = cps[i + 1];
                if (next_norm >= 0x30A1 && next_norm <= 0x30F3) {
                    next_norm = next_norm - 0x30A0 + 0x3040;
                }
                
                if (next_norm == 0x3083 || next_norm == 0x3085 || next_norm == 0x3087) { // ゃ, ゅ, ょ
                    // Check if base kana ends in 'i'
                    if (!romaji.empty() && romaji.back() == 'i') {
                        std::string base = romaji.substr(0, romaji.size() - 1);
                        std::string suffix;
                        if (next_norm == 0x3083) suffix = "ya";
                        else if (next_norm == 0x3085) suffix = "yu";
                        else suffix = "yo";
                        
                        // Irregular forms
                        if (base == "sh") {
                            if (next_norm == 0x3083) romaji = "sha";
                            else if (next_norm == 0x3085) romaji = "shu";
                            else romaji = "sho";
                        } else if (base == "ch") {
                            if (next_norm == 0x3083) romaji = "cha";
                            else if (next_norm == 0x3085) romaji = "chu";
                            else romaji = "cho";
                        } else if (base == "j") {
                            if (next_norm == 0x3083) romaji = "ja";
                            else if (next_norm == 0x3085) romaji = "ju";
                            else romaji = "jo";
                        } else {
                            romaji = base + suffix;
                        }
                        i++; // consume small kana
                    }
                }
            }

            if (double_next) {
                if (!romaji.empty()) {
                    out.push_back(romaji[0]);
                }
                double_next = false;
            }
            out += romaji;
        } else if (current == 0x30FC || current == 0x2015) { // Cho-onpu (ー) long vowel sign
            if (!out.empty()) {
                char last = out.back();
                if (last == 'a' || last == 'i' || last == 'u' || last == 'e' || last == 'o') {
                    out.push_back(last);
                }
            }
        } else {
            utf8_append(current, out);
        }
    }
    return out;
}

// ─── RUSSIAN (ru) ────────────────────────────────────────────────────────────

std::string cyrillic_to_latin_char(uint32_t cp) {
    switch (cp) {
        // Lowercase
        case 0x0430: return "a";
        case 0x0431: return "b";
        case 0x0432: return "v";
        case 0x0433: return "g";
        case 0x0434: return "d";
        case 0x0435: return "e";
        case 0x0451: return "yo";
        case 0x0436: return "zh";
        case 0x0437: return "z";
        case 0x0438: return "i";
        case 0x0439: return "y";
        case 0x043a: return "k";
        case 0x043b: return "l";
        case 0x043c: return "m";
        case 0x043d: return "n";
        case 0x043e: return "o";
        case 0x043f: return "p";
        case 0x0440: return "r";
        case 0x0441: return "s";
        case 0x0442: return "t";
        case 0x0443: return "u";
        case 0x0444: return "f";
        case 0x0445: return "kh";
        case 0x0446: return "ts";
        case 0x0447: return "ch";
        case 0x0448: return "sh";
        case 0x0449: return "shch";
        case 0x044a: return ""; // Hard sign
        case 0x044b: return "y";
        case 0x044c: return ""; // Soft sign
        case 0x044d: return "e";
        case 0x044e: return "yu";
        case 0x044f: return "ya";

        // Uppercase
        case 0x0410: return "a";
        case 0x0411: return "b";
        case 0x0412: return "v";
        case 0x0413: return "g";
        case 0x0414: return "d";
        case 0x0415: return "e";
        case 0x0401: return "yo";
        case 0x0416: return "zh";
        case 0x0417: return "z";
        case 0x0418: return "i";
        case 0x0419: return "y";
        case 0x041a: return "k";
        case 0x041b: return "l";
        case 0x041c: return "m";
        case 0x041d: return "n";
        case 0x041e: return "o";
        case 0x041f: return "p";
        case 0x0420: return "r";
        case 0x0421: return "s";
        case 0x0422: return "t";
        case 0x0423: return "u";
        case 0x0424: return "f";
        case 0x0425: return "kh";
        case 0x0426: return "ts";
        case 0x0427: return "ch";
        case 0x0428: return "sh";
        case 0x0429: return "shch";
        case 0x042a: return "";
        case 0x042b: return "y";
        case 0x042c: return "";
        case 0x042d: return "e";
        case 0x042e: return "yu";
        case 0x042f: return "ya";

        default: return "";
    }
}

std::string cyrillic_to_latin(const std::string& text) {
    std::vector<uint32_t> cps;
    size_t pos = 0;
    uint32_t cp;
    while (utf8_decode(text.data(), text.size(), pos, cp)) {
        cps.push_back(cp);
    }

    std::string out;
    for (auto c : cps) {
        std::string lat = cyrillic_to_latin_char(c);
        if (!lat.empty() || c == 0x044a || c == 0x044c || c == 0x042a || c == 0x042c) {
            out += lat;
        } else {
            utf8_append(c, out);
        }
    }
    return out;
}

// ─── HEBREW (he) ─────────────────────────────────────────────────────────────

std::string hebrew_to_latin_char(uint32_t cp) {
    switch (cp) {
        case 0x05D0: return "a"; // Alef
        case 0x05D1: return "v"; // Bet
        case 0x05D2: return "g"; // Gimel
        case 0x05D3: return "d"; // Dalet
        case 0x05D4: return "h"; // He
        case 0x05D5: return "v"; // Vav
        case 0x05D6: return "z"; // Zayin
        case 0x05D7: return "ch"; // Chet
        case 0x05D8: return "t"; // Tet
        case 0x05D9: return "y"; // Yod
        case 0x05DA: return "ch"; // Kaf Sofit
        case 0x05DB: return "k"; // Kaf
        case 0x05DC: return "l"; // Lamed
        case 0x05DD: return "m"; // Mem Sofit
        case 0x05DE: return "m"; // Mem
        case 0x05DF: return "n"; // Nun Sofit
        case 0x05E0: return "n"; // Nun
        case 0x05E1: return "s"; // Samekh
        case 0x05E2: return "a"; // Ayin
        case 0x05E3: return "f"; // Pe Sofit
        case 0x05E4: return "p"; // Pe
        case 0x05E5: return "ts"; // Tsadi Sofit
        case 0x05E6: return "ts"; // Tsadi
        case 0x05E7: return "k"; // Qof
        case 0x05E8: return "r"; // Resh
        case 0x05E9: return "sh"; // Shin
        case 0x05EA: return "t"; // Tav
        default: return "";
    }
}

std::string hebrew_to_latin(const std::string& text) {
    std::vector<uint32_t> cps;
    size_t pos = 0;
    uint32_t cp;
    while (utf8_decode(text.data(), text.size(), pos, cp)) {
        cps.push_back(cp);
    }

    std::string out;
    for (auto c : cps) {
        std::string lat = hebrew_to_latin_char(c);
        if (!lat.empty()) {
            out += lat;
        } else if (c >= 0x05B0 && c <= 0x05C7) {
            // Strip Nikud (vowel marks)
        } else {
            utf8_append(c, out);
        }
    }
    return out;
}

} // namespace

std::string preprocess_for_language(const std::string& text, const std::string& lang) {
    if (lang == "hi") {
        return devanagari_to_roman(text);
    } else if (lang == "ja") {
        return kana_to_romaji(text);
    } else if (lang == "ru") {
        return cyrillic_to_latin(text);
    } else if (lang == "he") {
        return hebrew_to_latin(text);
    }
    return text;
}

} // namespace tts_cpp::chatterbox::detail
