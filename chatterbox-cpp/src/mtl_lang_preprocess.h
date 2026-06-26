#pragma once

#include <string>

namespace tts_cpp::chatterbox::detail {

// Preprocesses text for a specific language (hi, ja, ru, he) before tokenization.
std::string preprocess_for_language(const std::string& text, const std::string& lang);

} // namespace tts_cpp::chatterbox::detail
