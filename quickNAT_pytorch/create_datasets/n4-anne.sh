export LD_LIBRARY_PATH="/home/abhijit/nas_drive/Software/ANTS/install/lib"
ANTSPATH=/home/abhijit/nas_drive/Software/ANTS/install/bin

INPUT_DIR="/home/abhijit/Jyotirmay/abdominal_segmentation/quickNAT_pytorch/create_datasets/temp/NAKO/n4_corrected/vol"
# for 1st run: *iso_in*rescaled*.nii.gz
# for 2nd run: *iso_in_corrected_comb_sigm_rescaled*

for file in "${INPUT_DIR}"/*;
do
echo "$file"

 for image in "${file}"/*n4_scaled*;
 do
 echo "$image"

 NAME=`echo "$image" | cut -d'.' -f1`
 echo "$NAME"
 ${ANTSPATH}/N4BiasFieldCorrection --bspline-fitting 400 \
 -d 3 \
 --input-image "${image}" \
 --output ["${NAME}"_corrected.nii.gz , "${NAME}"_bias_field.nii.gz] \
 --shrink-factor 3 \
 --convergence [500x500x500x500, 0.0001] \
 --verbose
 done
done
