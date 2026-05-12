# Import Required Libraries
import os
import matplotlib.pyplot as plt
import nibabel as nib


# Define the Function
def plot_image_matrix(list1, list2, row_labels, col_labels, save_path=None):
    """
    Plots images from two lists in a matrix, each list gets its own row and every entry is a column.
    """
    # Combine the lists into a single list of lists
    image_lists = [list1, list2]

    # Create the figure with tighter grid spacing
    plt.style.use("dark_background")
    fig, axes = plt.subplots(len(image_lists), len(list1), figsize=(2.5 * len(list1), 2.5 * len(image_lists)),
                             gridspec_kw={"wspace": 0.02, "hspace": 0.02})

    font_size = 14

    for row_idx, image_list in enumerate(image_lists):
        for col_idx, image_path in enumerate(image_list):
            # Load the image
            img = nib.load(image_path).get_fdata()

            # Plot the image
            ax = axes[row_idx, col_idx]
            ax.imshow(img[:, :, img.shape[2] // 2], cmap='gray', origin='lower')
            ax.axis('off')

            # Add row labels
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=font_size)

            # Add column labels
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=font_size)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


# Prediction map paths
base_path = r'C:\QTI_ML\runs\QTI_MLP_predict\BENCHMARK_ABSTRACT_prediction'
prefix = 'QTI_MLP_16P_bm_ABSTRACT_sub_min_case_m_ens_f18_ts_P18_vs_P1_20231029_14h_ep100_LR1e-03_bs512_outnorm_sched_92_avg_'

list1 = [
    os.path.join(base_path, 'P1', prefix + 'MD_pred.nii'),
    os.path.join(base_path, 'P1', prefix + 'FA_pred.nii'),
    os.path.join(base_path, 'P1', prefix + 'uFA_pred.nii'),
    os.path.join(base_path, 'P1', prefix + 'C_c_pred.nii'),
    os.path.join(base_path, 'P1', prefix + 'C_MD_pred.nii'),
]

list2 = [
    os.path.join(base_path, 'P14', prefix + 'MD_pred.nii'),
    os.path.join(base_path, 'P14', prefix + 'FA_pred.nii'),
    os.path.join(base_path, 'P14', prefix + 'uFA_pred.nii'),
    os.path.join(base_path, 'P14', prefix + 'C_c_pred.nii'),
    os.path.join(base_path, 'P14', prefix + 'C_MD_pred.nii'),
]

# Row and column labels
row_labels = ['P1', 'P14']
col_labels = ['MD', 'FA', 'uFA', 'C_c', 'C_MD']

save_path = os.path.join(base_path, 'QTI_MLP_prediction_maps_P1_P14.png')

# Call the function to plot the images
plot_image_matrix(list1, list2, row_labels, col_labels, save_path=save_path)
