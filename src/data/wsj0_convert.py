import argparse
import os
import subprocess


def main():

    # Input format is 'wv1' or 'wv2'

    # check if input directory exists
    if not os.path.exists(args.input):
        raise ValueError('input directory does not exist')

    # check if output directory exists
    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # create audio directory in output directory
    audio_dir = os.path.join(args.output, 'wsj0_audio')
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    # find all wv1 and wv2 files
    files_sphere_format = []
    for root, folders, files in os.walk(args.input):
        for file in filter(lambda x: x.endswith('.wv1') or x.endswith('.wv2'), files):
            files_sphere_format.append(os.path.join(root, file))
    n_files = len(files_sphere_format)

    # main loop
    for i, filepath in enumerate(files_sphere_format):

        # show progress
        if not args.quiet:
            print(f'{i}/{n_files} ({i/n_files*100:.1f}%)')

        # Recreate the same folder structure, but the *.wv1 files are converted to *.wav files
        # The folder structure is the same as the original WSJ0 corpus

        # create output directory
        outfile = os.path.join(audio_dir, os.path.relpath(filepath, args.input))
        outfile_extension = os.path.splitext(outfile)[-1]
        outfile_wav = outfile.replace(outfile_extension, '.wav')

        output_directory = os.path.dirname(outfile)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if os.path.exists(outfile_wav):
            if not args.quiet:
                print(f'{outfile_wav} already exists')
            continue

        # call sph2pipe to convert to wav
        if not args.quiet:
            print(f'writing {outfile_wav}')
        subprocess.call([
            'sph2pipe',
            '-f',
            'wav',
            filepath,
            outfile_wav,
        ])


if __name__ == '__main__':
    default_wsj0_path = '/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets/wsj0'
    default_output_path = '/Users/giovannibologni/Documents/TU-Delft/Code-parent/datasets'

    parser = argparse.ArgumentParser(description='convert WSJ0 corpus')
    parser.add_argument('--input', help='path to WSJ0 corpus', default=default_wsj0_path)
    parser.add_argument('--output', help='output directory', default=default_output_path)
    parser.add_argument('--quiet', action='store_true', help='disable verbose')
    args = parser.parse_args()
    main()
