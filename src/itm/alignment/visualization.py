import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from itm_schema.ml_pipeline import AlignmentPackage
from itm_schema.kdma_ids import KDMAId
from itm.alignment.similarity_functions import _kde_to_pdf


def plot_alignment(
        package: AlignmentPackage,
        kdma_id: KDMAId,
        target_id: str,
        ax=None,
        add_overlay: bool =False,
        samples: int = 100,
        file_name: str = None,
    ) -> None:

    if ax is None:
        ax=plt.figure(figsize=(8, 6)).gca()

    kdma_name = kdma_id

    # Target KDE
    kde1 = package.alignment_target.target[target_id].kdma_measurements[kdma_id].kde
    kde1_dm_id = target_id

    # Aligner KDE
    kde2 = package.aligner_profile.kdma_measurements[kdma_id].kde
    kde2_dm_id = package.aligner_id

    dist = package.overall_alignment

    x = np.linspace(0, 1, samples)
    pdf_kde1 = _kde_to_pdf(kde1, x)
    pdf_kde2 = _kde_to_pdf(kde2, x)

    # Plot the KDEs and shade the overlap region
    kde1color = 'black'
    kde2color = 'blue'
    ax.plot(x, pdf_kde1, label='KDE 1', color=kde1color)
    ax.plot(x, pdf_kde2, label='KDE 2', color=kde2color)

    # Calculate the overlap area
    overlap_area = np.trapz(np.minimum(pdf_kde1, pdf_kde2), x)

    # Shade the overlap region
    ax.fill_between(x, np.minimum(pdf_kde1, pdf_kde2), color='gray', alpha=0.5)

    ax.set_xlabel(f'{kdma_name}')
    ax.set_ylabel('Probability Density')
    ax.tick_params(axis='both')
    ax.set_title(
        f'{kde1_dm_id}({kde1color}) vs {kde2_dm_id} ({kde2color})\nAlignment (JS): {dist:.4f}'
    )
    #ax.legend()

    if add_overlay:
        #print("Adding overlay")
        # Create a colormap with a gradient from one end to the other based on dist
        custom_colors_hex = ['#FF0000', '#FFFB00', '70FF00'] # Red/Yellow/Green
        custom_colors_rgb = [tuple(int(color[i:i+2], 16) / 255.0 for i in (1, 3, 5)) for color in custom_colors_hex]
        n_bins = 1000
        cmap_name = 'custom_gradient'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, custom_colors_rgb, N=n_bins)

        # Add a color overlay to the entire graph using the dist value
        rgba_color = custom_cmap(dist)
        ax.imshow([[rgba_color]], aspect='auto', extent=(0, 1, 0, max(pdf_kde1.max(), pdf_kde2.max())), alpha=.3)

        # Overlay JS Divergence value in large letters at the center of the image
        # To make the overlay text always appear at the center of the image, dynamically calculate the vertical position 
        # based on the maximum values of the two KDEs. Then set the vertical position of the overlay text (text_vertical_position) 
        # to half of this maximum value. This way, the text should be centered vertically on the plot, regardless of the actual values of the KDEs.

        # Calculate the maximum value among the two KDEs (to help compute where to put the text overlay)
        max_kde_value = max(pdf_kde1.max(), pdf_kde2.max())

        # Overlay JS Divergence value in large letters at the center of the image
        text_vertical_position = max_kde_value / 2  # You can adjust this based on your preference
        ax.text(0.5, text_vertical_position, f'{dist:.4f}', fontsize=72, ha='center', color='black', backgroundcolor='none')

    # Save the figure for each profile if a filename was given
    if file_name != None:
        plt.savefig(file_name, bbox_inches='tight', dpi=600)
        plt.close()
