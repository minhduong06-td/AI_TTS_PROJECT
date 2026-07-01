from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.utils.text.characters import IPAPhonemes

from vits_common import (
    build_audio_config,
    build_dataset_config,
    build_vits_config,
    parse_resume_args,
    run_training,
)

args = parse_resume_args()


def build_vi_characters_config() -> CharactersConfig:
    base_vi_ipa = IPAPhonemes()
    vi_characters = base_vi_ipa.characters + "1234567̪"
    vi_punctuations = base_vi_ipa.punctuations + "«»“”"

    return CharactersConfig(
        characters=vi_characters,
        punctuations=vi_punctuations,
        pad=base_vi_ipa.pad,
        eos=base_vi_ipa.eos,
        bos=base_vi_ipa.bos,
        blank=base_vi_ipa.blank,
        is_unique=True,
        is_sorted=True,
    )

LANG_TAG = "vi"
DATASET_NAME = "vi_ljspeech"
META_FILE = "metadata_clean.csv"
DATASET_PATH = f"/home/md_dz6/AI/data/{DATASET_NAME}"

RUN_NAME = "vi_ljspeech_vits_real"
OUTPUT_PATH = f"/home/md_dz6/AI/coqui_runs/{LANG_TAG}/{RUN_NAME}"

TEXT_CLEANER = "vietnamese_cleaners"
PHONEMIZER = "espeak"
PHONEME_LANGUAGE = "vi"
SAMPLE_RATE = 24000

TEST_SENTENCES = [
    ["Xin chào, tôi đang học cách đọc tiếng Việt rõ ràng hơn."],
    ["Hôm nay trời nắng nhẹ, gió thổi chậm và không khí rất dễ chịu."],
    ["Tôi thích đọc sách, nghe nhạc, và học thêm những điều mới mỗi ngày."],
    ["Một chiếc lá rơi xuống mặt nước, rồi lặng lẽ trôi đi."],
    ["Câu này dài hơn một chút, để kiểm tra xem mô hình có biết ngắt nghỉ tự nhiên hay không."],
    ["Bạn cần đọc rõ các âm cuối như học, sách, thích, lạnh, mạnh, lặng, mát và mặc."],
]

VI_CHARACTERS = build_vi_characters_config()

dataset_config = build_dataset_config(
    dataset_name=DATASET_NAME,
    dataset_path=DATASET_PATH,
    meta_file=META_FILE,
    language=LANG_TAG,
)

audio_config = build_audio_config(
    sample_rate=SAMPLE_RATE,
)

config = build_vits_config(
    run_name=RUN_NAME,
    output_path=OUTPUT_PATH,
    dataset_config=dataset_config,
    audio_config=audio_config,
    text_cleaner=TEXT_CLEANER,
    phonemizer=PHONEMIZER,
    phoneme_language=PHONEME_LANGUAGE,
    test_sentences=TEST_SENTENCES,
    characters=VI_CHARACTERS,
)


config.lr_gen = 1e-4                
config.lr_disc = 2e-5              
config.mel_loss_alpha = 60.0         
config.dur_loss_alpha = 1.3         
config.disc_loss_alpha = 0.5         
config.gen_loss_alpha = 1.0          
config.feat_loss_alpha = 1.0         
config.kl_loss_alpha = 1.0          
config.batch_group_size = 5          

config.min_audio_len = SAMPLE_RATE * 1      
config.max_audio_len = SAMPLE_RATE * 14     
config.min_text_len = 4                    
config.max_text_len = 260                

config.save_step = 2000             
config.save_n_checkpoints = 30      
config.save_all_best = True          
config.save_best_after = 2000        
config.target_loss = "loss_mel"      

config.model_args.inference_noise_scale = 0.35      
config.model_args.inference_noise_scale_dp = 0.6    

run_training(
    config=config,
    dataset_config=dataset_config,
    output_path=OUTPUT_PATH,
    continue_path=args.continue_path,
    restore_path=args.restore_path,
)
