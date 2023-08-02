#!/bin/bash

# Set the directory containing the image directories
base_dir="/robodata/arthurz/Datasets/CODa_bbox_full/2d_raw/cam0"
out_dir="/robodata/arthurz/Media/CODa_bbox"

## Uncomment to generate egocomp point cloud segmentation
# base_dir="/robodata/arthurz/Datasets/CODa_sem_full/2d_raw/cam0"
# out_dir='/robodata/arthurz/Media/CODa_sem'

## Uncomment to generate non egocomp point cloud segmentation
# base_dir="/robodata/arthurz/Datasets/CODa_sem_nocomp_full/2d_raw/cam0"
# out_dir='/robodata/arthurz/Media/CODa_sem_nocomp'

# Loop through each subdirectory in the base directory
for dir in "${base_dir}"/*; do
    if [[ -d "${dir}" ]]; then
        # Extract the directory name
        dir_name=$(basename "${dir}")

        # # Create the output video filename
        output_video="${out_dir}/${dir_name}.mp4"
        echo "Current image directory ${dir}"
        # # Use ffmpeg to generate the video from images
        ffmpeg -framerate 20 -pattern_type glob -i "${dir}/2d_rect_cam0_${dir_name}_*.jpg" -c:v libx264 "${output_video}"
        # ffmpeg -framerate 20 -i "${dir}/%d.jpg" -c:v libx264 -pix_fmt yuv420p "${output_video}"
        echo "Video generated: ${output_video}"

        # Compress generated video
        # compressed_output_video="${out_dir}/comp${dir_name}.mp4"
        # ffmpeg -i "${output_video}" -vcodec libx264 -crf 28 "${compressed_output_video}"

        echo "Video generated: ${compressed_output_video}"
    fi
done

echo "All videos generated successfully."
