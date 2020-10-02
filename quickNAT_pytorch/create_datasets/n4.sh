export LD_LIBRARY_PATH="/Users/jyotirmaysenapati/Projects/PYTHON/abdominal_segmentation/quickNAT_pytorch/create_datasets/install/lib"
ANTSPATH=/Users/jyotirmaysenapati/Projects/PYTHON/abdominal_segmentation/quickNAT_pytorch/create_datasets/install/bin

INPUT_DIR="/Users/jyotirmaysenapati/Projects/PYTHON/abdominal_segmentation/quickNAT_pytorch/create_datasets/temp/kora_in_volume.nii.gz"

#"/home/anne/whole_body_segmentation/All_KORA_original_data/All/"
# for 1st run: *iso_in*rescaled*.nii.gz
# for 2nd run: *iso_in_corrected_comb_sigm_rescaled*
for file in "${INPUT_DIR}"*;
do
echo "$file"

#for image in "${file}"/*iso_in_corrected_comb_sigm_rescaled*;
#do
#echo "$image"

NAME=`echo "$file" | cut -d'.' -f1`
echo "$NAME"
${ANTSPATH}/N4BiasFieldCorrection --bspline-fitting 400 \
-d 3 \
--input-image "${file}" \
--output ["${NAME}"_corrected.nii.gz , "${NAME}"_bias_field.nii.gz] \
--shrink-factor 3 \
--convergence [200x200x200x200, 0.0001] \
--verbose 
done
