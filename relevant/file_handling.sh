#!/usr/bin/bash

TARGET_FOLDER="/home/hck03/Documents/MS_thesis/tracking_output/1c_4n_faulty"

for folder in "$TARGET_FOLDER"/*/
do
    [ -e "$folder" ] || continue    
    # echo "${folder}"
    target_subfolder="${folder}data"
    if [ -d "$target_subfolder" ]; then
        # echo "$target_subfolder"
        shopt -s nullglob
        for npz_file in "$target_subfolder"/*.npz; do
            file_name=$(basename "$npz_file")
            echo "$filename"

            # FOOLPROOF NAME STRIPPING: Removes the last extension regardless of casing (.npz or .NPZ)
            output_dir_name="${file_name%.*}"
            
            # Fallback check: If it somehow still resolves to blank, name it after the file safely
            if [ -z "$output_dir_name" ]; then
                output_dir_name="extracted_${file_name}"
            fi
            
            output_dir="${target_subfolder}/${output_dir_name}"
            
            echo "  --> Unzipping to: $output_dir_name"
            
            mkdir -p "$output_dir"
            unzip -q -o "$npz_file" -d "$output_dir"
                
        done
        shopt -u nullglob
    fi
done
