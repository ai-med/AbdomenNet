export LD_LIBRARY_PATH="/home/abhijit/nas_drive/Software/ANTS/install/lib"
ANTSPATH=/home/abhijit/nas_drive/Software/ANTS/install/bin

INPUT_DIR="/home/abhijit/Jyotirmay/abdominal_segmentation/quickNAT_pytorch/create_datasets/temp/KORA/n4_corrected/vol"

for file in "${INPUT_DIR}"/*;
do
echo "$file"

 for image in "${file}"/*IN_ras_stitched_n4_scaled.nii.gz;
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
