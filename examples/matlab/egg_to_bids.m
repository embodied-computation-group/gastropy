% egg_to_bids.m  --  Convert EGG data to BIDS physio format for GastroPy
%
% This script converts a multi-channel EGG recording into the BIDS
% peripheral physiology format:
%
%   _physio.tsv.gz   gzip-compressed, tab-separated, no header row
%   _physio.json     JSON sidecar with SamplingFrequency, StartTime, Columns
%
% After conversion, the files can be loaded directly with:
%
%   from gastropy.io import read_bids_physio
%   data = read_bids_physio("sub-01_task-rest_physio.tsv.gz")
%
% BIDS specification reference:
%   https://bids-specification.readthedocs.io/en/stable/modality-specific-files/
%   physiological-and-other-continuous-recordings.html
%
% Requirements: MATLAB R2016b+ (for jsonencode)
%
% Usage:
%   1. Edit the "User parameters" section below
%   2. Run the script
%   3. Copy the output files into your BIDS dataset or pass directly to GastroPy
%
% Author: GastroPy contributors
% License: MIT

%% ========================================================================
%  User parameters -- edit these for your data
%  ========================================================================

% Path to your EGG data (adapt the loading code below to your format)
input_file = 'my_egg_recording.mat';

% BIDS entities
subject   = '01';           % subject ID (without "sub-" prefix)
session   = '';             % session ID (leave empty to omit)
task      = 'rest';         % task label

% Recording parameters
sfreq     = 10.0;           % sampling frequency in Hz (after any downsampling)
start_time = 0.0;           % recording start time relative to task onset (s)

% Channel names -- one per EGG channel, in column order
ch_names  = {'EGG1', 'EGG2', 'EGG3', 'EGG4', 'EGG5', 'EGG6', 'EGG7'};

% Output directory
output_dir = '.';

%% ========================================================================
%  Load your EGG data
%  ========================================================================
%  Adapt this section to your acquisition system. The result should be:
%
%    signal   (n_channels x n_samples) double matrix
%    sfreq    sampling frequency in Hz
%    ch_names cell array of channel name strings
%
%  Examples for common formats:

% --- Example 1: MATLAB .mat file ---
% S = load(input_file);
% signal = S.EGG;              % (n_channels x n_samples)

% --- Example 2: BrainVision (via EEGLAB) ---
% EEG = pop_loadbv(folder, 'recording.vhdr');
% signal = EEG.data;           % (n_channels x n_samples)
% sfreq  = EEG.srate;
% ch_names = {EEG.chanlocs.labels};

% --- Example 3: CSV / text file ---
% data = readmatrix('recording.csv');  % (n_samples x n_channels)
% signal = data';                       % transpose to (n_channels x n_samples)

% For this template, generate a small test signal:
fprintf('WARNING: Using synthetic test data. Replace this with your actual data loading code.\n');
n_samples = round(600 * sfreq);  % 10 minutes
n_channels = numel(ch_names);
t = (0:n_samples-1) / sfreq;
signal = zeros(n_channels, n_samples);
for ch = 1:n_channels
    signal(ch, :) = sin(2 * pi * 0.05 * t + ch * 0.3) + 0.1 * randn(1, n_samples);
end

%% ========================================================================
%  Optional: downsample to target frequency
%  ========================================================================

% target_sfreq = 10.0;
% if sfreq > target_sfreq
%     [p, q] = rat(target_sfreq / sfreq);
%     signal = resample(signal', p, q)';  % resample operates on columns
%     sfreq = target_sfreq;
%     n_samples = size(signal, 2);
%     fprintf('Downsampled to %.1f Hz (%d samples)\n', sfreq, n_samples);
% end

%% ========================================================================
%  Build BIDS filename
%  ========================================================================

parts = {sprintf('sub-%s', subject)};
if ~isempty(session)
    parts{end+1} = sprintf('ses-%s', session);
end
parts{end+1} = sprintf('task-%s', task);
parts{end+1} = 'physio';
bids_stem = strjoin(parts, '_');

tsv_path  = fullfile(output_dir, [bids_stem, '.tsv.gz']);
json_path = fullfile(output_dir, [bids_stem, '.json']);

%% ========================================================================
%  Write _physio.tsv.gz
%  ========================================================================
%  BIDS physio TSV rules:
%    - No header row
%    - Tab-separated columns
%    - One row per time point
%    - One column per channel
%    - Gzip-compressed (extension .tsv.gz)

% First write uncompressed TSV to a temp file, then gzip it
tsv_tmp = fullfile(output_dir, [bids_stem, '.tsv']);

fid = fopen(tsv_tmp, 'w');
if fid == -1
    error('Could not open %s for writing', tsv_tmp);
end

% Transpose: (n_channels x n_samples) -> write row-by-row as (n_samples x n_channels)
fmt = repmat('%.10g\t', 1, n_channels);
fmt = [fmt(1:end-1), '\n'];  % replace trailing tab with newline

for i = 1:size(signal, 2)
    fprintf(fid, fmt, signal(:, i));
end
fclose(fid);

% Gzip the TSV file
gzip(tsv_tmp);
delete(tsv_tmp);  % remove uncompressed version

fprintf('Wrote %s\n', tsv_path);

%% ========================================================================
%  Write _physio.json sidecar
%  ========================================================================
%  Required BIDS fields:
%    SamplingFrequency  (Hz)
%    StartTime          (seconds, relative to task onset)
%    Columns            (cell array of channel names)

metadata = struct();
metadata.SamplingFrequency = sfreq;
metadata.StartTime         = start_time;
metadata.Columns           = ch_names;

% Optional fields -- add as needed:
% metadata.Description = 'EGG recording during resting state';
% metadata.Manufacturer = 'BioSemi';
% metadata.ManufacturersModelName = 'ActiveTwo';

json_str = jsonencode(metadata);

% Pretty-print the JSON (basic indentation)
json_str = strrep(json_str, ',"', sprintf(',\n    "'));
json_str = strrep(json_str, '{', sprintf('{\n    '));
json_str = strrep(json_str, '}', sprintf('\n}'));

fid = fopen(json_path, 'w');
fprintf(fid, '%s\n', json_str);
fclose(fid);

fprintf('Wrote %s\n', json_path);

%% ========================================================================
%  Verify output
%  ========================================================================

fprintf('\nDone. Files ready for GastroPy:\n');
fprintf('  %s\n', tsv_path);
fprintf('  %s\n', json_path);
fprintf('\nIn Python:\n');
fprintf('  from gastropy.io import read_bids_physio\n');
fprintf('  data = read_bids_physio("%s")\n', tsv_path);
fprintf('  print(data["signal"].shape)  # (%d, %d)\n', n_channels, size(signal, 2));
fprintf('  print(data["sfreq"])         # %.1f\n', sfreq);
