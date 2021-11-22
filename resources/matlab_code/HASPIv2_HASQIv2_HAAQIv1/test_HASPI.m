user_guide_root = ...
    'G:\My Drive\DTU\Kurser\Deep_Learning_02456\final_project\resources\code\HASPIv2_HASQIv2_HAAQIv1\user_guide';

path_signal_clean = fullfile(user_guide_root, 'sig_clean.wav');
path_signal_out = fullfile(user_guide_root, 'sig_out.wav');

[signal_clean, fs_signal_clean] = audioread(path_signal_clean);
[signal_out, fs_signal_out] = audioread(path_signal_out);

audiometric_frequencies = [250, 500, 1000, 2000, 4000, 6000];
HL = [0, 0, 10, 20, 40, 30];

% sound(signal_clean, fs)

[intelligibility, raw] = HASPI_v2(...
    signal_clean, ...
    fs_signal_clean, ...
    signal_out, ...
    fs_signal_out, ...
    HL);
