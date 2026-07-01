def inv_spectrogram(postnet_output, ap, CONFIG):
    if CONFIG.model.lower() in ["tacotron"]:
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_melspectrogram(postnet_output.T)
    return wav


def apply_griffin_lim(inputs, input_lens, CONFIG, ap):
    wavs = []
    for idx, spec in enumerate(inputs):
        wav_len = (input_lens[idx] * ap.hop_length) - ap.hop_length  
        wav = inv_spectrogram(spec, ap, CONFIG)
        wavs.append(wav[:wav_len])
    return wavs
